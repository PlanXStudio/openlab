import usb.core
import usb.util

VID = 0x2886
PID = 0x0018

class FoundNotPixelRing(Exception): pass

hex2rgb = lambda n: ((n >> 16) & 0xFF, (n >> 8) & 0xFF, n & 0xFF)
rgb2invert = lambda r, g, b: (~r & 0xFF, ~g & 0xFF, ~b & 0xFF)

class PixelRing:

    def __init__(self, brightness=10):
        """AudioPixelRing 객체 생성

        Args:
            brightness (int): LED 밝기 (0 ~ 20)
            None: 기본 값 10 (중간 밝기)

        Returns:
            None
        """

        self.dev = usb.core.find(idVendor=VID, idProduct=PID)
        if not self.dev:
            raise FoundNotPixelRing
        self.brightness(brightness)

    def __del__(self):
        """AudioPixelRing 객체가 제거될 때 USB 자원 해제

        Args:
            None

        Returns:
            None
        """

        usb.util.dispose_resources(self.dev)
        del self.dev
    
    def __write(self, cmd, data=[0]):
        self.dev.ctrl_transfer(
                usb.util.CTRL_OUT | 
                usb.util.CTRL_TYPE_VENDOR | 
                usb.util.CTRL_RECIPIENT_DEVICE,
                0, cmd, 0x1C, data, 8000
                )
    
    write = __write

    def __palette(self, a, b):
        r1, g1, b1, r2, g2, b2 = *(hex2rgb(a) if (not type(a) is tuple) else a), *(hex2rgb(b) if (not type(b) is tuple) else b) 
        print("###", r1, g1, b1, r2, g2, b2)
        self.__write(0x21, [r1, g1, b1, 0, r2, g2, b2, 0])

    def normal(self, *color):
        """단일 색상 출력

        Args:
            color (hex): 16진수 RGB 색상 값 (0x000000 ~ 0xFFFFFFFF)
            color (int, int, int): 10진수 R, G, B 색상 값 (0 ~ 255, 0~255, 0~255) 

        Returns:
            None
        """

        r, g, b = hex2rgb(*color) if len(color) == 1 else color 
        self.__write(1, [r, g, b, 0])

    def off(self):
        """출력 중지

        Args:
            None

        Returns:
            None
        """

        self.normal(0x000000)
    
    def listen(self, *palette):
        """전경과 배경 색을 기반으로 소리 감지 방향 출력
        
        Args:
            palette (hex1, hex2): 16진수 RGB 감지 위치(hex1), 배경(hex2) 색상 값
            palette (r1, g1, b1, r2, g2, b2): 10진수 감지 위치(R1, G1, B1), 배경( R2, G2, B2) 색상 값
            None: 이전 값 사용 

        Returns:
            None
        """

        self.__write(0)
        if palette:
            self.__palette(*palette)

    def aurora(self, *palette):
        """2가지 색을 조합해 오로라 패턴으로 출력

        Args:
            palette (hex1, hex2): 16진수 RGB 시작(hex1), 종료(hex2) 색상 값
            palette (r1, g1, b1, r2, g2, b2): 10진수 시작(R1, G1, B1),종료( R2, G2, B2) 색상 값
            None: 이전 값 사용

        Returns
            None
        """
        
        self.__write(3)
        if palette:
            self.__palette(*palette)

    def think(self, *palette):
        """2가지 색으로 짝수, 홀수 바꿔가며 출력

        Args:
            palette (ex1, hex2): 16진수 RGB 시작(hex1), 종료(hex2) 색상 값
            palette (r1, g1, b1, r2, g2, b2): 10진수 시작(R1, G1, B1),종료( R2, G2, B2) 색상 값
            None: 이전 값 사용

        Returns
            None      
        """
      
        self.__write(4)
        if palette:
            self.__palette(*palette)

    def spin(self, *color):
        """한 가지 색의 밝기를 차례로 조절하며 스핀 패턴으로 출력

        Args:
            color (hex): 16진수 RGB 색상 값 (0x000000 ~ 0xFFFFFFFF):
            color (int, int, int): 10진수 R, G, B 색상 값 (0 ~ 255, 0~255, 0~255)
            None: 이전 값 사용
        
        Returns:
            None
        """

        self.__write(5)
        if color:
            color = (color[0], ~color[0]) if len(color) == 1 else (color, rgb2invert(*color))
            self.__palette(*color)

    def brightness(self, val): # do not color()
        """밝기 설정

        Args:
            val (int): 밝기 (0 ~ 20)

        Returns:
            None
        """

        self.__write(0x20, [val])
