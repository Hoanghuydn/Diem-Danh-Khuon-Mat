from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image, ImageDraw
from datetime import *
import time
from math import *


class Login_window:
    def __init__(self, root):
        self.root = root
        self.root.title("GUI Analog Clock")
        self.root.geometry("1350x700+0+0")
        self.root.config(bg="#021e2f")

        # title=Label(self.root,text="Welcome Analog Clock",font=("times new roman",50,"bold"),bg="#04444a",fg="white").place(x=0,y=50,relwidth=1)
        # left_lbl=Label(self.root,bg="#08A3D2",bd=0)
        # left_lbl.place(x=0,y=0,relheight=1,width=600)

        # right_lbl=Label(self.root,bg="#031F3C",bd=0)
        # right_lbl.place(x=600,y=0,relheight=1,relwidth=1)

        Frame_login = Frame(self.root, bg="white")
        Frame_login.place(x=150, y=150, height=340, width=600)

        self.lbl = Label(self.root, bg="white", bd=0)
        self.lbl.place(x=700, y=120, height=450, width=350)
        # self.clock_image()
        self.working()

    def clock_image(self, hr, min_, sec_):
        clock = Image.new("RGB", (400, 400), (255, 255, 255))
        draw = ImageDraw.Draw(clock)
        # =====For Clock Image
        bg = Image.open("test/images/clock.jpg")
        bg = bg.resize((300, 300), Image.ANTIALIAS)
        clock.paste(bg, (50, 50))

        # Formula To Rotate the AntiClock
        # angle_in_radians = angle_in_degress * math.pi / 180
        # line_length = 100
        # center_x = 250
        # center_y = 250
        # end_x = center_x + line_length * math.cos(angle_in_radians)
        # end_y = center_y - line_length * math.sin(angle_in_radians) # + nguoc chieu - cung chieu

        # ========= Hour Line Image====
        #           x1,y1,x2,y2  250,280,300
        origin = 200, 200
        draw.line(
            (origin, 200 + 50 * sin(radians(hr)), 200 - 50 * cos(radians(hr))),
            fill="black",
            width=4,
        )
        # ========= Min Line Image====
        draw.line(
            (origin, 200 + 80 * sin(radians(min_)), 200 - 80 * cos(radians(min_))),
            fill="blue",
            width=3,
        )
        # ========= Sec Line Image====
        draw.line(
            (origin, 200 + 100 * sin(radians(sec_)), 200 - 100 * cos(radians(sec_))),
            fill="green",
            width=4,
        )
        draw.ellipse((195, 195, 210, 210), fill="black")

        clock.save("test/images/clock_new.png")

    def working(self):
        h = datetime.now().time().hour
        m = datetime.now().time().minute
        s = datetime.now().time().second
        # print(h,m,s)
        hr = (h / 12) * 360
        min_ = (m / 60) * 360
        sec_ = (s / 60) * 360
        # print(hr,min_,sec_)
        self.clock_image(hr, min_, sec_)
        self.img = ImageTk.PhotoImage(file="test/images/clock_new.png")
        self.lbl.config(image=self.img)
        self.lbl.after(200, self.working)


root = Tk()
obj = Login_window(root)
root.mainloop()
