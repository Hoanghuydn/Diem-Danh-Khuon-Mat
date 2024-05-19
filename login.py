from tkinter import *
from tkinter import messagebox
from PIL import ImageTk, Image, ImageDraw
from datetime import *
import time
from math import *
import pymysql


class Login:
    def __init__(self, root):
        self.root = root
        self.root.title("Login System")
        self.root.geometry("1190x600+100+50")
        self.root.resizable(True, True)
        # =====BG Image=====
        #self.bg = ImageTk.PhotoImage(file=)
        self.bg = ImageTk.PhotoImage(file="images/login.jpg")
        self.bg_image = Label(self.root, image=self.bg).place(x=0, y=0, relwidth=1, relheight=1)
        # =====Login Image=====
        Frame_login = Frame(self.root, bg="white")
        Frame_login.place(x=150, y=150, height=340, width=600)

        title = Label(Frame_login, text="Login Here", font=("Impact", 35, "bold"), fg="#d77337", bg="white").place(x=90,
                                                                                                                   y=30)

        lbl_user = Label(Frame_login, text="Username", font=("Goudy old style", 15, "bold"), fg="gray",
                         bg="white").place(x=90, y=140)
        self.txt_user = Entry(Frame_login, font=("times new roman", 15), bg="lightgray")
        self.txt_user.place(x=90, y=170, width=350, height=35)

        lbl_pass = Label(Frame_login, text="Password", font=("Goudy old style", 15, "bold"), fg="gray",
                         bg="white").place(x=90, y=210)
        self.txt_pass = Entry(Frame_login, font=("times new roman", 15), bg="lightgray")
        self.txt_pass.place(x=90, y=240, width=350, height=35)

        # =========bd=0 an nut forget_btn=Button(Frame_login,text="Forget Password?",cursor="hand2",bg="white",
        # fg="#d77337",bd=0,font=("times new roman",12)).place(x=300,y=280)
        Login_btn = Button(self.root, command=self.login, cursor="hand2", text="Login", fg="white", bg="#d77337",
                           font=("times new roman", 20)).place(x=300, y=470, width=180, height=40)
        btn_reg = Button(Frame_login, text="Register new Account?", command=self.register_window, cursor="hand2",
                         font=("times new roman", 12), bg="white", fg="#d77337", bd=0).place(x=80, y=280)

        # ==========clock=====x trai phai y tren xuong
        self.lbl = Label(self.root, bg="#000000", bd=0)
        self.lbl.place(x=650, y=120, height=420, width=350)
        # self.clock_image()
        self.working()

    def register_window(self):
        self.root.destroy()
        import register1

    def login(self):
        # print(self.txt_user.get(),self.txt_pass.get())
        if self.txt_user.get() == "" or self.txt_pass.get() == "":
            messagebox.showerror("Error", "All fields are required", parent=self.root)
        else:
            try:
                con = pymysql.connect(host="localhost", user="root", password="", database="hoag")
                # duyệt qua các bản ghi trong tập dữ liệu được lấy về và thực thi các câu truy vấn
                # Để thực thi một câu truy vấn thì chúng ta dùng phương thức execute().
                cur = con.cursor()
                cur.execute("select * from data where email=%s and password=%s",
                            (self.txt_user.get(), self.txt_pass.get()))
                # Phương thức fetchone() lấy về dòng đầu tiên của bảng dữ liệu trả về.
                row = cur.fetchone()
                # print(row)
                if row == None:
                    messagebox.showerror("Error", "Invalid USERNAME & PASSWORD", parent=self.root)
                else:
                    messagebox.showinfo("Success", "Welcome " + str(self.txt_user.get()), parent=self.root)
                    self.root.destroy()
                    import background
                con.close()
            except Exception as es:
                messagebox.showerror("Error", f"Error due to: {str(es)}", parent=self.root)

    def clock_image(self, hr, min_, sec_):
        clock = Image.new("RGB", (400, 400), (0, 0, 0))
        draw = ImageDraw.Draw(clock)
        # =====For Clock Image
        bg = Image.open("images/b.jpg")
        #bg = Image.open("images/b.jpg")
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
        draw.line((origin, 200 + 50 * sin(radians(hr)), 200 - 50 * cos(radians(hr))), fill="white", width=4)
        # ========= Min Line Image====
        draw.line((origin, 200 + 80 * sin(radians(min_)), 200 - 80 * cos(radians(min_))), fill="blue", width=3)
        # ========= Sec Line Image====
        draw.line((origin, 200 + 100 * sin(radians(sec_)), 200 - 100 * cos(radians(sec_))), fill="green", width=4)
        draw.ellipse((195, 195, 210, 210), fill="yellow")
        clock.save("images/clock_new.png")
        #clock.save("images/clock_new.png")

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
        self.img = ImageTk.PhotoImage(file="images/clock_new.png")
        self.lbl.config(image=self.img)
        self.lbl.after(200, self.working)

    # def login_function(self):
    #     if self.txt_pass.get()=="" or self.txt_user.get()=="":
    #         messagebox.showerror("Error","All fields are required",parent=self.root)
    #     elif self.txt_pass.get()!="123" or self.txt_user.get()!="lan":
    #         messagebox.showerror("Error","Invalid Username/Password",parent=self.root)
    #     else:
    #         messagebox.showinfo("Welcome",f"Welcome {self.txt_user.get()}\nYour Password: {self.txt_pass.get()}",parent=self.root)


root = Tk()
obj = Login(root)
root.mainloop()
