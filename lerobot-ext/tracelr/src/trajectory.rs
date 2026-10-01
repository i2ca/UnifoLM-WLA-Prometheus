use std::collections::{HashMap, VecDeque};
use std::path::{Path, PathBuf};

use arrow::array::Array;

/// Computed EE trajectory for a single episode.
#[derive(Clone)]
pub(crate) struct EeTrajectory {
    /// Per-frame end-effector position [x, y, z] in meters (the first/primary end effector).
    pub positions: Vec<[f64; 3]>,
    /// Other end effectors of the same robot (e.g. the LEFT hand of a humanoid), same frames.
    pub extra: Vec<Vec<[f64; 3]>>,
}

/// One serial chain (base -> end effector) and, for each of its movable joints, where its angle is in
/// `observation.state` (by NAME when the dataset names match the URDF, otherwise by order).
struct Cadeia {
    nome: String,
    serial: k::SerialChain<f64>,
    /// index in the state row for each movable joint of `serial` (None = keep at 0)
    indices: Vec<Option<usize>>,
    /// state values are degrees (`.pos`, SO101-style) and must be converted to radians
    graus: bool,
}

/// Robot kinematics context loaded from a URDF file.
/// Wraps the `k` crate's serial chain(s) for FK computation.
pub(crate) struct RobotKinematics {
    cadeias: Vec<Cadeia>,
}

/// "kLeftShoulderPitch.q" (Unitree SDK) and "left_shoulder_pitch_joint" (URDF) -> "leftshoulderpitch".
fn normaliza(nome: &str) -> String {
    let mut s = nome.trim();
    for suf in [".q", ".pos"] {
        if let Some(x) = s.strip_suffix(suf) {
            s = x;
        }
    }
    let mut c = s.chars();
    let sem_k = match (c.next(), c.next()) {
        (Some('k'), Some(u)) if u.is_ascii_uppercase() => &s[1..],
        _ => s,
    };
    let mut n: String = sem_k.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    if let Some(x) = n.strip_suffix("joint") {
        n = x.to_string();
    }
    n
}

impl RobotKinematics {
    /// Load a URDF and build the kinematic chain(s).
    /// - `ee_frame` Some: that link only.
    /// - URDF with `left_wrist_yaw_link` and `right_wrist_yaw_link` (Unitree G1/H1): TWO chains, right
    ///   hand first, left hand second, ending at `*_hand_palm_link` when it exists.
    /// - otherwise: the deepest leaf link.
    /// `state_names` (the dataset's `observation.state` names) lets the joints be matched BY NAME;
    /// joints the dataset does not have stay at 0. If no joint matches, falls back to the old by-order
    /// behaviour (`.pos`/`.q` arm columns, see `pos_indices_from_state_names`).
    #[allow(dead_code)]
    pub fn from_urdf(urdf_path: &Path, ee_frame: Option<&str>) -> Result<Self, String> {
        Self::from_urdf_with_state(urdf_path, ee_frame, &[])
    }

    pub fn from_urdf_with_state(urdf_path: &Path, ee_frame: Option<&str>, state_names: &[String]) -> Result<Self, String> {
        let chain = k::Chain::<f64>::from_urdf_file(urdf_path)
            .map_err(|e| format!("Failed to load URDF {}: {}", urdf_path.display(), e))?;

        let mut fins: Vec<(String, k::node::Node<f64>)> = Vec::new();
        if let Some(name) = ee_frame {
            let n = chain.find_link(name).ok_or_else(|| format!("EE frame '{}' not found in URDF", name))?;
            fins.push((name.to_string(), n.clone()));
        } else if chain.find_link("right_wrist_yaw_link").is_some() && chain.find_link("left_wrist_yaw_link").is_some() {
            for lado in ["right", "left"] {
                let palma = format!("{}_hand_palm_link", lado);
                let punho = format!("{}_wrist_yaw_link", lado);
                let n = chain.find_link(&palma).or_else(|| chain.find_link(&punho)).unwrap();
                fins.push((format!("{} hand", lado), n.clone()));
            }
        } else {
            fins.push(("ee".to_string(), find_deepest_leaf(&chain)?));
        }

        let por_nome: HashMap<String, usize> = state_names
            .iter()
            .enumerate()
            .filter(|(_, n)| n.ends_with(".q") || n.ends_with(".pos"))
            .map(|(i, n)| (normaliza(n), i))
            .collect();
        let graus = state_names.iter().any(|n| n.ends_with(".pos")) && !state_names.iter().any(|n| n.ends_with(".q"));
        let por_ordem = pos_indices_from_state_names(state_names);

        let mut cadeias = Vec::new();
        for (nome, fim) in fins {
            let serial = k::SerialChain::from_end(&fim);
            let juntas: Vec<String> = serial
                .iter()
                .filter(|n| !matches!(n.joint().joint_type, k::joint::JointType::Fixed))
                .map(|n| n.joint().name.clone())
                .collect();
            let mut indices: Vec<Option<usize>> = juntas.iter().map(|j| por_nome.get(&normaliza(j)).copied()).collect();
            let achou = indices.iter().filter(|i| i.is_some()).count();
            if achou == 0 {
                // old behaviour: the arm columns, in order
                indices = (0..juntas.len()).map(|k| por_ordem.get(k).copied()).collect();
            }
            log::info!(
                "Loaded URDF: {} -> {} (DOF={}, {} of {} joints matched by name: {:?})",
                urdf_path.display(), nome, serial.dof(), achou, juntas.len(),
                juntas.iter().zip(&indices).map(|(j, i)| format!("{}={}", j, i.map(|v| v.to_string()).unwrap_or("-".into()))).collect::<Vec<_>>(),
            );
            cadeias.push(Cadeia { nome, serial, indices, graus });
        }
        Ok(Self { cadeias })
    }

    fn fk(c: &Cadeia, state: &[f32]) -> [f64; 3] {
        let positions: Vec<f64> = c
            .indices
            .iter()
            .map(|i| {
                let v = i.and_then(|i| state.get(i)).copied().unwrap_or(0.0) as f64;
                if c.graus { v.to_radians() } else { v }
            })
            .collect();
        // Use unchecked — real robot data can slightly exceed URDF limits
        c.serial.set_joint_positions_unchecked(&positions);
        c.serial.update_transforms();
        let t = c.serial.end_transform().translation;
        [t.x, t.y, t.z]
    }

    /// Compute the EE trajectory (every chain) for a full episode of joint states.
    /// `_pos_indices` is kept for compatibility: the joint -> column map is built at load time.
    pub fn compute_trajectory(&self, states: &[Vec<f32>], _pos_indices: &[usize]) -> EeTrajectory {
        let mut todas: Vec<Vec<[f64; 3]>> = self
            .cadeias
            .iter()
            .map(|c| states.iter().map(|s| Self::fk(c, s)).collect())
            .collect();
        let positions = if todas.is_empty() { Vec::new() } else { todas.remove(0) };
        EeTrajectory { positions, extra: todas }
    }

    pub fn dof(&self) -> usize {
        self.cadeias.first().map(|c| c.serial.dof()).unwrap_or(0)
    }

    /// Names of the end effectors, primary first ("right hand", "left hand" on the G1).
    pub fn ee_names(&self) -> Vec<String> {
        self.cadeias.iter().map(|c| c.nome.clone()).collect()
    }
}

/// Load `observation.state` from a parquet data file.
/// For v3.0 shared files, filters rows by `episode_index`.
/// `filter_episode` should be Some(idx) for v3.0, None for v2.1 (entire file is one episode).
pub(crate) fn load_episode_states(parquet_path: &Path, filter_episode: Option<usize>) -> Result<Vec<Vec<f32>>, String> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let file = std::fs::File::open(parquet_path)
        .map_err(|e| format!("Open {}: {}", parquet_path.display(), e))?;

    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|e| format!("Parquet reader {}: {}", parquet_path.display(), e))?;

    let reader = builder.build()
        .map_err(|e| format!("Build reader {}: {}", parquet_path.display(), e))?;

    let mut all_states: Vec<Vec<f32>> = Vec::new();

    for batch in reader {
        let batch = batch.map_err(|e| format!("Read batch: {}", e))?;

        // For v3.0: filter rows by episode_index column
        let row_mask: Option<Vec<bool>> = if let Some(target_ep) = filter_episode {
            if let Some(ep_col) = batch.column_by_name("episode_index") {
                let ep_arr = ep_col
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .ok_or_else(|| "episode_index is not Int64Array".to_string())?;
                Some((0..ep_arr.len()).map(|i| ep_arr.value(i) as usize == target_ep).collect())
            } else {
                None
            }
        } else {
            None
        };

        let state_col = batch
            .column_by_name("observation.state")
            .ok_or_else(|| "No 'observation.state' column in parquet".to_string())?;

        let list_arr = state_col
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
            .ok_or_else(|| "observation.state is not FixedSizeListArray".to_string())?;

        let values = list_arr
            .values()
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()
            .ok_or_else(|| "observation.state values are not Float32".to_string())?;

        let list_size = list_arr.value_length() as usize;
        for i in 0..list_arr.len() {
            if let Some(ref mask) = row_mask {
                if !mask[i] {
                    continue;
                }
            }
            let offset = i * list_size;
            let row: Vec<f32> = (0..list_size).map(|j| values.value(offset + j)).collect();
            all_states.push(row);
        }
    }

    Ok(all_states)
}

/// Build the parquet data path for an episode.
/// v2.1: `data/chunk-NNN/episode_NNNNNN.parquet` (one file per episode)
/// v3.0: `data/chunk-NNN/file-NNN.parquet` (shared files, need to filter by episode_index)
pub(crate) fn episode_data_path(dataset_root: &Path, episode_index: usize, chunks_size: usize, codebase_version: &str) -> PathBuf {
    let chunk = episode_index / chunks_size;
    if codebase_version.starts_with("v3") {
        // v3.0: episodes are packed into shared files. Find the right file.
        // Each file holds `chunks_size` episodes, file index = episode_index / chunks_size within chunk.
        // For simplicity, scan for files in the chunk dir.
        let chunk_dir = dataset_root
            .join("data")
            .join(format!("chunk-{:03}", chunk));
        // v3.0 uses file-NNN.parquet; episode_index within chunk determines file
        let file_idx = episode_index % chunks_size;
        // Actually in v3.0, all episodes in a chunk may be in a single file or split.
        // Try file-000 first (most common for small datasets).
        let path = chunk_dir.join(format!("file-{:03}.parquet", 0));
        if path.is_file() {
            return path;
        }
        // Fallback: try matching file index
        chunk_dir.join(format!("file-{:03}.parquet", file_idx))
    } else {
        dataset_root
            .join("data")
            .join(format!("chunk-{:03}", chunk))
            .join(format!("episode_{:06}.parquet", episode_index))
    }
}

/// LRU cache of computed EE trajectories, keyed by episode index.
pub(crate) struct TrajectoryCache {
    entries: HashMap<usize, EeTrajectory>,
    order: VecDeque<usize>,
    capacity: usize,
}

impl TrajectoryCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::new(),
            order: VecDeque::new(),
            capacity,
        }
    }

    pub fn get(&mut self, episode_index: usize) -> Option<&EeTrajectory> {
        if self.entries.contains_key(&episode_index) {
            self.order.retain(|&i| i != episode_index);
            self.order.push_back(episode_index);
            self.entries.get(&episode_index)
        } else {
            None
        }
    }

    pub fn insert(&mut self, episode_index: usize, trajectory: EeTrajectory) {
        if self.entries.contains_key(&episode_index) {
            self.order.retain(|&i| i != episode_index);
        } else if self.entries.len() >= self.capacity {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        self.entries.insert(episode_index, trajectory);
        self.order.push_back(episode_index);
    }
}

/// Find the deepest leaf link in a kinematic chain (most ancestors).
/// Used to auto-detect the end-effector frame.
fn find_deepest_leaf(chain: &k::Chain<f64>) -> Result<k::node::Node<f64>, String> {
    let mut best: Option<(k::node::Node<f64>, usize)> = None;
    for node in chain.iter() {
        if node.children().is_empty() {
            // Leaf node — count depth by walking parents
            let mut depth = 0;
            let mut cur = node.clone();
            while let Some(parent) = cur.parent() {
                depth += 1;
                cur = parent;
            }
            if best.as_ref().is_none_or(|&(_, d)| depth > d) {
                best = Some((node.clone(), depth));
            }
        }
    }
    best.map(|(n, _)| n)
        .ok_or_else(|| "No leaf links found in URDF".to_string())
}

/// Extract the indices of `.pos` values from state feature names.
/// e.g. ["joint_1.pos", "joint_1.vel", "joint_1.torque", "joint_2.pos", ...]
/// Returns indices of names ending in ".pos", excluding "gripper".
pub(crate) fn pos_indices_from_state_names(state_names: &[String]) -> Vec<usize> {
    state_names
        .iter()
        .enumerate()
        .filter(|(_, name)| {
            let is_joint = name.ends_with(".pos") || name.ends_with(".q");

            let is_arm = name.contains("Shoulder")
                || name.contains("Elbow")
                || name.contains("Wrist");

            is_joint && is_arm
        })
        .map(|(i, _)| i)
        .collect()
}

/// Discover the URDF file for a dataset.
/// Search order:
/// 1. `<dataset_root>/robot.urdf`
/// 2. `~/.config/tracelr/robots/<robot_type>.urdf`
/// 3. Bundled paths for known robots
pub(crate) fn discover_urdf(dataset_root: &Path, robot_type: Option<&str>) -> Option<PathBuf> {
    // 1. Dataset-local
    let local = dataset_root.join("robot.urdf");
    if local.is_file() {
        return Some(local);
    }

    // 2. User config directory
    if let Some(rt) = robot_type {
        if let Some(config_dir) = dirs::config_dir() {
            let user_urdf = config_dir
                .join("tracelr")
                .join("robots")
                .join(format!("{}.urdf", rt));
            if user_urdf.is_file() {
                return Some(user_urdf);
            }
        }
    }

    None
}

#[cfg(test)]
mod testes_g1 {
    use super::*;

    /// FK do G1 por NOME (colunas "kLeftShoulderPitch.q" do SDK da Unitree) contra a referência do MuJoCo:
    /// TRACELR_FK_REF=<arquivo com "right x y z", "left x y z" e "STATE nome=valor ..."> cargo test
    #[test]
    fn g1_duas_maos_batem_com_mujoco() {
        let Ok(ref_path) = std::env::var("TRACELR_FK_REF") else { return };
        let txt = std::fs::read_to_string(ref_path).unwrap();
        let mut esperado: HashMap<String, [f64; 3]> = HashMap::new();
        let (mut nomes, mut valores) = (Vec::new(), Vec::new());
        for linha in txt.lines() {
            let mut it = linha.split_whitespace();
            match it.next() {
                Some("STATE") => {
                    for kv in it {
                        let (n, v) = kv.split_once('=').unwrap();
                        nomes.push(n.to_string());
                        valores.push(v.parse::<f32>().unwrap());
                    }
                }
                Some(l) => {
                    let v: Vec<f64> = it.map(|x| x.parse().unwrap()).collect();
                    esperado.insert(format!("{} hand", l), [v[0], v[1], v[2]]);
                }
                None => {}
            }
        }
        let urdf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/g1/g1_body29_hand14.urdf");
        let kin = RobotKinematics::from_urdf_with_state(&urdf, None, &nomes).unwrap();
        let tr = kin.compute_trajectory(&[valores], &[]);
        let todas = [tr.positions[0], tr.extra[0][0]];
        for (nome, p) in kin.ee_names().iter().zip(todas) {
            let e = esperado[nome];
            let erro = ((p[0] - e[0]).powi(2) + (p[1] - e[1]).powi(2) + (p[2] - e[2]).powi(2)).sqrt();
            println!("{nome}: tracelr {:?} mujoco {:?} erro {:.4} m", p, e, erro);
            assert!(erro < 0.005, "{nome}: erro {erro} m");
        }
    }
}

#[cfg(test)]
mod teste_dataset_g1 {
    use super::*;

    /// TRACELR_DATASET=<dataset LeRobot do G1> cargo test dataset_g1_duas_maos -- --nocapture
    #[test]
    fn dataset_g1_duas_maos() {
        let Ok(raiz) = std::env::var("TRACELR_DATASET") else { return };
        let ds = crate::dataset::LeRobotDataset::load(Path::new(&raiz)).unwrap();
        let urdf = Path::new(env!("CARGO_MANIFEST_DIR")).join("../assets/g1/g1_body29_hand14.urdf");
        let kin = RobotKinematics::from_urdf_with_state(&urdf, None, &ds.info.state_names).unwrap();
        let v3 = ds.info.codebase_version.starts_with("v3");
        let p = episode_data_path(&ds.root, 0, ds.info.chunks_size, &ds.info.codebase_version);
        let st = load_episode_states(&p, if v3 { Some(0) } else { None }).unwrap();
        let tr = kin.compute_trajectory(&st, &[]);
        let mut todas = vec![tr.positions.clone()];
        todas.extend(tr.extra.clone());
        for (nome, pos) in kin.ee_names().iter().zip(&todas) {
            let (a, b) = (pos.first().unwrap(), pos.last().unwrap());
            let mut lo = [f64::MAX; 3];
            let mut hi = [f64::MIN; 3];
            for q in pos { for i in 0..3 { lo[i] = lo[i].min(q[i]); hi[i] = hi[i].max(q[i]); } }
            println!("{nome}: {} quadros | início {:.3?} fim {:.3?} | amplitude cm [{:.1}, {:.1}, {:.1}]",
                pos.len(), a, b, (hi[0]-lo[0])*100.0, (hi[1]-lo[1])*100.0, (hi[2]-lo[2])*100.0);
        }
        assert_eq!(todas.len(), 2);
    }
}
