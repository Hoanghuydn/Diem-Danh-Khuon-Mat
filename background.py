from tkinter import *
from PIL import ImageTk
from tkinter import filedialog
from tkinter import ttk, messagebox
import pymysql
import time
from datetime import datetime
import os
from tkinter import Toplevel

class Hover_Effect:
    def __init__(self, root):
        self.root = root
        self.root.title("")
        self.root.geometry("1020x700+250+50")
        self.root.config(bg="#262626")

        title = Label(
            self.root,
            text="Face Attandance Management System",
            bd=10,
            relief=GROOVE,
            font=("times new roman", 30, "bold"),
        )
        title.pack(side=TOP, fill=X)
        self.employee = ImageTk.PhotoImage(file="images/em.png")
        lbl_em = Button(
            self.root, image=self.employee, bd=0, cursor="hand2", command=self.open_stu
        )
        lbl_em.place(x=30, y=90, width=300, height=300)

        # self.fol = ImageTk.PhotoImage(file="images/folder.png")
        # lbl_fol = Button(
        #     self.root, image=self.fol, bd=0, cursor="hand2", command=self.open_img
        # )
        # lbl_fol.place(x=360, y=90, width=300, height=300)

        # self.train = ImageTk.PhotoImage(file="images/brain.jpg")
        # lbl_tr = Button(
        #     self.root, image=self.train, bd=0, cursor="hand2", command=self.open_tra
        # )
        # lbl_tr.place(x=690, y=90, width=300, height=300)

        self.fac = ImageTk.PhotoImage(file="images/facere.png")
        lbl_fac = Button(
            self.root, image=self.fac, bd=0, cursor="hand2", command=self.open_reco
        )
        lbl_fac.place(x=360, y=90, width=300, height=300)

        self.shee = ImageTk.PhotoImage(file="images/shee.png")
        lbl_shee = Button(
            self.root, image=self.shee, bd=0, cursor="hand2", command=self.open_excel
        )
        lbl_shee.place(x=30, y=410, width=300, height=280)

        self.pas = ImageTk.PhotoImage(file="images/chpa.png")
        lbl_pas = Button(
            self.root, image=self.pas, bd=0, cursor="hand2", command=self.open_chpass
        )
        lbl_pas.place(x=360, y=410, width=300, height=280)

        self.dev = ImageTk.PhotoImage(file="images/deve.png")
        lbl_dev = Button(
            self.root, image=self.dev, bd=0, cursor="hand2", command=self.open_develop
        )
        lbl_dev.place(x=690, y=410, width=300, height=280)

        self.exit = ImageTk.PhotoImage(file="images/exit.png")
        lbl_ex = Button(
            self.root, image=self.exit, bd=0, cursor="hand2", command=self.open_exit
        )
        #lbl_ex.place(x=1020, y=410, width=300, height=280)
        lbl_ex.place(x=690, y=90, width=300, height=300)

    def open_stu(self):
        import Student

    # def open_img(self):
    #     self.filename = filedialog.askopenfilename(
    #         initialdir="datasets",
    #         title="File",
    #         filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")),
    #     )

    def open_tra(self):
        import train_model

    def open_reco(self):
        import detect_face

    def open_excel(self):
        import xlwrite

        os.startfile(
            os.getcwd() + "/excel/attendance" + str(datetime.now().date()) + ".xls"
        )

    def open_chpass(self):
        self.root2 = Toplevel()
        self.root2.title("Change Password")
        self.root2.geometry("350x400+495+150")
        self.root2.config(bg="white")
        self.root2.focus_force()
        self.root2.grab_set()

        t = Label(
            self.root2,
            text="Change Password",
            font=("times new roman", 20, "bold"),
            bg="white",
            fg="red",
        ).place(x=0, y=10, relwidth=1)

        user = Label(
            self.root2,
            text="Username",
            font=("times new roman", 15, "bold"),
            fg="gray",
            bg="white",
        ).place(x=50, y=100)
        self.txt_user1 = Entry(self.root2, font=("times new roman", 15), bg="lightgray")
        self.txt_user1.place(x=50, y=130, width=250)

        old_pass = Label(
            self.root2,
            text="Old Password",
            font=("times new roman", 15, "bold"),
            bg="white",
            fg="gray",
        ).place(x=50, y=180)
        self.old_pass = Entry(self.root2, font=("times new roman", 15), bg="lightgray")
        self.old_pass.place(x=50, y=210, width=250)

        new_pass = Label(
            self.root2,
            text="New Password",
            font=("times new roman", 15, "bold"),
            bg="white",
            fg="gray",
        ).place(x=50, y=260)
        self.txt_new_pass = Entry(
            self.root2, font=("times new roman", 15), bg="lightgray"
        )
        self.txt_new_pass.place(x=50, y=290, width=250)

        btn_change_pass = Button(
            self.root2,
            text="Reset Password",
            bg="green",
            fg="white",
            command=self.change,
            font=("times new roman", 15, "bold"),
        ).place(x=90, y=340)

    def change(self):
        if self.txt_user1.get() == "" or self.old_pass.get() == "":
            messagebox.showerror("Error", "All fields are required", parent=self.root2)
        else:
            try:
                con = pymysql.connect(
                    host="localhost", user="root", password="", database="hoag"
                )
                cur = con.cursor()
                cur.execute(
                    "select * from data where email=%s and password=%s",
                    (self.txt_user1.get(), self.old_pass.get()),
                )
                row = cur.fetchone()
                if row == None:
                    messagebox.showerror(
                        "Error", "Invalid USERNAME & PASSWORD", parent=self.root
                    )
                else:
                    cur.execute(
                        "update data set password=%s where email=%s",
                        (self.txt_new_pass.get(), self.txt_user1.get()),
                    )
                    con.commit()
                    con.close()
                    messagebox.showinfo(
                        "Success",
                        "Your password has been reset,Please login with new password",
                        parent=self.root2,
                    )
                    self.root.destroy()
                    import login
            except Exception as es:
                messagebox.showerror(
                    "Error", f"Error due to: {str(es)}", parent=self.root2
                )

    def open_develop(self):
        messagebox.showinfo(
            "About",
            "version v1.0\n Made Using\n-OpenCV\n-Numpy\n-Tkinter\n-Pillow\n-Xlwrite In Python 3",
        )

    def open_exit(self):
        self.root.destroy()


root = Tk()
obj = Hover_Effect(root)
root.mainloop()
