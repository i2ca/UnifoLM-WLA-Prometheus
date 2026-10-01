"""Standalone sensor utilities for camera image publishing via ZMQ"""
import base64
import json
from dataclasses import dataclass
from typing import Any, Dict

import cv2
import numpy as np
import zmq


@dataclass
class ImageMessageSchema:
    """
    Standardized message schema for image data.
    Used to serialize/deserialize image data for network transmission.
    """

    timestamps: Dict[str, float]
    """Dictionary of timestamps, keyed by image identifier (e.g., {"ego_view": 123.45})"""
    images: Dict[str, np.ndarray]
    """Dictionary of images, keyed by image identifier (e.g., {"ego_view": array})"""

    def serialize(self) -> Dict[str, Any]:
        """Serialize the message for transmission."""
        serialized_msg = {"timestamps": self.timestamps, "images": {}}
        for key, image in self.images.items():
            serialized_msg["images"][key] = ImageUtils.encode_image(image)
        return serialized_msg

    @staticmethod
    def deserialize(data: Dict[str, Any]) -> "ImageMessageSchema":
        """Deserialize received message data."""
        timestamps = data.get("timestamps", {})
        images = {}
        for key, value in data.get("images", {}).items():
            if isinstance(value, str):
                images[key] = ImageUtils.decode_image(value)
            else:
                images[key] = value
        return ImageMessageSchema(timestamps=timestamps, images=images)

    def asdict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {"timestamps": self.timestamps, "images": self.images}


class SensorServer:
    """ZMQ-based sensor server for publishing camera images"""
    
    def start_server(self, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.SNDHWM, 20)  # high water mark
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.bind(f"tcp://*:{port}")
        print(f"Sensor server running at tcp://*:{port}")

        self.message_sent = 0
        self.message_dropped = 0

    def stop_server(self):
        self.socket.close()
        self.context.term()

    def send_message(self, data: Dict[str, Any], parts: list[bytes] | None = None):
        """Publica uma mensagem, opcionalmente com quadros binários extras.

        `parts` carrega o que não cabe em JSON — profundidade uint16, por
        exemplo. Cada imagem assim vai em `data["images"]` como um dicionário
        que aponta para o quadro: `{"part": 1, "encoding": "png"}` (ou
        `{"part": 1, "dtype": "uint16", "shape": [H, W]}` para buffer cru).
        Índice 0 é sempre o JSON.

        Sem `parts` a mensagem sai com um quadro só, byte a byte igual à do
        `send_string` antigo — consumidor velho não vê diferença.
        """
        try:
            frames = [json.dumps(data).encode("utf-8")]
            if parts:
                frames.extend(parts)
            self.socket.send_multipart(frames, flags=zmq.NOBLOCK)
        except zmq.Again:
            self.message_dropped += 1
            print(f"[Warning] message dropped: {self.message_dropped}")
        self.message_sent += 1

        if self.message_sent % 100 == 0:
            print(
                f"[Sensor server] Message sent: {self.message_sent}, message dropped: {self.message_dropped}"
            )


class SensorClient:
    """ZMQ-based sensor client for subscribing to camera images"""
    
    def start_client(self, server_ip: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        # ZMQ_CONFLATE saiu daqui: ele é INCOMPATÍVEL com mensagem multiparte.
        # Com ele ligado o socket entrega uma parte só — e nem é a primeira:
        # chega o PNG da profundidade sem o JSON, e o json.loads estoura. O
        # efeito que ele dava (ficar só com o quadro mais novo) está
        # reimplementado no receive_message, esvaziando a fila na mão.
        self.socket.setsockopt(zmq.RCVHWM, 3)  # queue size 3 for receive buffer
        self.socket.connect(f"tcp://{server_ip}:{port}")

    def stop_client(self):
        self.socket.close()
        self.context.term()

    def receive_message(self):
        """Recebe uma mensagem e já resolve as imagens que vieram em quadro binário.

        `recv_multipart` também lê a mensagem de um quadro só do formato antigo,
        então isto continua servindo servidor velho. As imagens que vieram como
        dicionário (`{"part": ...}`) voltam já como `np.ndarray`; as em base64
        continuam como string, decodificadas pelo chamador como sempre.
        """
        frames = self.socket.recv_multipart()
        # Esvazia o que ficou para trás: numa teleoperação o quadro atrasado não
        # serve para nada, e acumular fila vira latência que só cresce.
        while True:
            try:
                frames = self.socket.recv_multipart(zmq.NOBLOCK)
            except zmq.Again:
                break
        data = json.loads(frames[0].decode("utf-8"))
        images = data.get("images")
        if isinstance(images, dict):
            for name, payload in images.items():
                if isinstance(payload, dict):
                    images[name] = ImageUtils.decode_part(payload, frames)
        return data


class ImageUtils:
    """Utilities for encoding/decoding images for network transmission"""
    
    @staticmethod
    def encode_image(image: np.ndarray) -> str:
        """Encode numpy image to base64-encoded JPEG string"""
        _, color_buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        return base64.b64encode(color_buffer).decode("utf-8")

    @staticmethod
    def encode_depth_image(image: np.ndarray) -> str:
        """Encode depth image to base64-encoded PNG string"""
        depth_compressed = cv2.imencode(".png", image)[1].tobytes()
        return base64.b64encode(depth_compressed).decode("utf-8")

    @staticmethod
    def encode_raw(image: np.ndarray, part: int) -> tuple:
        """Descritor + buffer CRU da imagem, sem compressão nenhuma.

        Para profundidade (uint16) não dá para usar o caminho do RGB: JPEG é
        lossy e só vai a 8 bits. PNG de 16 bits resolveria a fidelidade, mas
        medido no próprio robô custa ~20 ms por quadro — dois terços do
        orçamento de 33 ms a 30 fps, com o RGB e o align da RealSense ainda
        para pagar. Cru custa zero de CPU e são 814 KB por quadro (24 MB/s a
        30 fps), o que é ~20% de um enlace gigabit; robô e PC estão os dois em
        eth0 a 1000 Mb/s, então a rede é o recurso de sobra aqui, não a CPU.

        Devolve `(descritor, bytes)`: o descritor vai dentro do JSON, em
        `data["images"][nome]`, e os bytes viram um quadro binário da mensagem.
        """
        buffer = np.ascontiguousarray(image)
        descritor = {"part": part, "dtype": buffer.dtype.str, "shape": list(buffer.shape)}
        return descritor, buffer.tobytes()

    @staticmethod
    def decode_part(payload: Dict[str, Any], frames: list[bytes]) -> np.ndarray:
        """Resolve uma imagem que veio em quadro binário separado.

        Mesmo protocolo que o `_decode_zmq_images` do fork do LeRobot lê
        (`lerobot/cameras/zmq/camera_zmq.py`) — as duas pontas precisam
        concordar, porque o mesmo servidor alimenta a teleoperação e a gravação.
        """
        index = payload["part"]
        if index >= len(frames):
            raise RuntimeError(f"mensagem sem o quadro binário {index}")
        buffer = frames[index]

        if "dtype" in payload:  # buffer cru, sem compressão
            array = np.frombuffer(buffer, dtype=np.dtype(payload["dtype"]))
            return array.reshape(payload["shape"]).copy()

        # IMREAD_UNCHANGED e não IMREAD_COLOR: é ele que preserva os 16 bits e o
        # canal único da profundidade.
        flags = cv2.IMREAD_UNCHANGED if payload.get("encoding") == "png" else cv2.IMREAD_COLOR
        image = cv2.imdecode(np.frombuffer(buffer, np.uint8), flags)
        if image is None:
            raise RuntimeError("falha ao decodificar imagem de quadro binário")
        return image

    @staticmethod
    def decode_image(image) -> np.ndarray:
        """Decode base64-encoded JPEG string to numpy image"""
        # O `receive_message` já entrega array pronto quando a imagem veio em
        # quadro binário. Devolver como está deixa o chamador chamar isto sempre.
        if isinstance(image, np.ndarray):
            return image
        color_data = base64.b64decode(image)
        color_array = np.frombuffer(color_data, dtype=np.uint8)
        return cv2.imdecode(color_array, cv2.IMREAD_COLOR)

    @staticmethod
    def decode_depth_image(image: str) -> np.ndarray:
        """Decode base64-encoded PNG string to depth image"""
        depth_data = base64.b64decode(image)
        depth_array = np.frombuffer(depth_data, dtype=np.uint8)
        return cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)

