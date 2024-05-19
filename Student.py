from tkinter import *
from tkinter import ttk
import pymysql
from tkinter import messagebox
import cv2
import os
import numpy as np
from tkinter import Toplevel

class Student:
    def __init__(self, root):
        self.root = root
        self.root.title("Student Management System")
        self.root.geometry("1350x700+0+0")

        title = Label(self.root, text="Student Management System", bd=10, relief=GROOVE,
                      font=("times new roman", 40, "bold"), bg="#6550c7", fg="white")
        title.pack(side=TOP, fill=X)


        
        # Handle the window close event
        self.root.wm_protocol("WM_DELETE_WINDOW", self.on_close)
    
        self.Roll_No_var = StringVar()
        self.name_var = StringVar()
        self.email_var = StringVar()
        self.gender_var = StringVar()
        self.contact_var = StringVar()
        self.dob_var = StringVar()
        self.search_by = StringVar()
        self.search_txt = StringVar()

        # Text khong nhan Stringvar nen address ko nhan

        # ===========manage frame===========

        Manage_Frame = Frame(self.root, bd=4, relief=RIDGE, bg="#6550c7")
        Manage_Frame.place(x=20, y=100, width=450, height=590)

        m_title = Label(Manage_Frame, text="Manage Students", bg="#6550c7", fg="white",
                        font=("times new roman", 30, "bold"))
        m_title.grid(row=0, columnspan=2, pady=5)

        lbl_roll = Label(Manage_Frame, text="Roll No.", bg="#6550c7", fg="white", font=("times new roman", 20, "bold"))
        lbl_roll.grid(row=1, column=0, pady=10, padx=20, sticky="w")

        self.txt_Roll = Entry(Manage_Frame, textvariable=self.Roll_No_var, font=("times new roman", 15, "bold"), bd=5,
                              relief=GROOVE)
        self.txt_Roll.grid(row=1, column=1, pady=10, padx=20, sticky="w")

        lbl_name = Label(Manage_Frame, text="Name", bg="#6550c7", fg="white", font=("times new roman", 20, "bold"))
        lbl_name.grid(row=2, column=0, pady=10, padx=20, sticky="w")

        self.txt_name = Entry(Manage_Frame, textvariable=self.name_var, font=("times new roman", 15, "bold"), bd=5,
                              relief=GROOVE)
        self.txt_name.grid(row=2, column=1, pady=10, padx=20, sticky="w")

        lbl_Email = Label(Manage_Frame, text="Email", bg="#6550c7", fg="white", font=("times new roman", 20, "bold"))
        lbl_Email.grid(row=3, column=0, pady=10, padx=20, sticky="w")

        self.txt_Email = Entry(Manage_Frame, textvariable=self.email_var, font=("times new roman", 15, "bold"), bd=5,
                               relief=GROOVE)
        self.txt_Email.grid(row=3, column=1, pady=10, padx=20, sticky="w")

        lbl_Gender = Label(Manage_Frame, text="Gender", bg="#6550c7", fg="white", font=("times new roman", 20, "bold"))
        lbl_Gender.grid(row=4, column=0, pady=10, padx=20, sticky="w")

        self.combo_gender = ttk.Combobox(Manage_Frame, textvariable=self.gender_var,
                                         font=("times new roman", 13, "bold"), state='readonly')
        self.combo_gender['values'] = ("male", "female", "other")
        self.combo_gender.grid(row=4, column=1, padx=20, pady=10)

        lbl_Contact = Label(Manage_Frame, text="Contact", bg="#6550c7", fg="white",
                            font=("times new roman", 20, "bold"))
        lbl_Contact.grid(row=5, column=0, pady=10, padx=20, sticky="w")

        self.txt_Contact = Entry(Manage_Frame, textvariable=self.contact_var, font=("times new roman", 15, "bold"),
                                 bd=5, relief=GROOVE)
        self.txt_Contact.grid(row=5, column=1, pady=10, padx=20, sticky="w")

        lbl_Dob = Label(Manage_Frame, text="D.O.B", bg="#6550c7", fg="white", font=("times new roman", 20, "bold"))
        lbl_Dob.grid(row=6, column=0, pady=10, padx=20, sticky="w")

        self.txt_Dob = Entry(Manage_Frame, textvariable=self.dob_var, font=("times new roman", 15, "bold"), bd=5,
                             relief=GROOVE)
        self.txt_Dob.grid(row=6, column=1, pady=10, padx=20, sticky="w")

        lbl_Address = Label(Manage_Frame, text="Address", bg="#6550c7", fg="white",
                            font=("times new roman", 20, "bold"))
        lbl_Address.grid(row=7, column=0, pady=10, padx=20, sticky="w")

        self.txt_Address = Text(Manage_Frame, width=30, height=4, font=("", 10))
        self.txt_Address.grid(row=7, column=1, pady=10, padx=20, sticky="w")

        # ==========button frame=========
        btn_Frame = Frame(Manage_Frame, bd=4, relief=RIDGE, bg="#6550c7")
        btn_Frame.place(x=15, y=490, width=420, height=90)

        Addbtn = Button(btn_Frame, text="Add", width=10, command=self.add_students).grid(row=0, column=0, padx=10,
                                                                                         pady=10)
        updatebtn = Button(btn_Frame, text="Update", command=self.update_data, width=10).grid(row=0, column=1, padx=10,
                                                                                              pady=10)
        deletebtn = Button(btn_Frame, text="Delete", command=self.delete_data, width=10).grid(row=0, column=2, padx=10,
                                                                                              pady=10)
        Clearbtn = Button(btn_Frame, text="Clear", command=self.clear, width=10).grid(row=0, column=3, padx=10, pady=10)

        imgbtn = Button(btn_Frame, text="Add photo", command=self.add_photo, width=28).place(x=100, y=50)

        Detail_Frame = Frame(self.root, bd=4, relief=RIDGE, bg="#6550c7")
        Detail_Frame.place(x=500, y=100, width=800, height=590)

        lbl_search = Label(Detail_Frame, text="Search_By", bg="#6550c7", fg="white",
                           font=("times new roman", 20, "bold"))
        lbl_search.grid(row=0, column=0, pady=10, padx=20, sticky="w")

        combo_search = ttk.Combobox(Detail_Frame, textvariable=self.search_by, width=10,
                                    font=("times new roman", 13, "bold"), state='readonly')
        combo_search['values'] = ("Roll_no", "Name", "Contact")
        combo_search.grid(row=0, column=1, padx=20, pady=10)

        txt_Search = Entry(Detail_Frame, textvariable=self.search_txt, width=20, font=("times new roman", 10, "bold"),
                           bd=5, relief=GROOVE)
        txt_Search.grid(row=0, column=2, pady=10, padx=20, sticky="w")

        searchbtn = Button(Detail_Frame, text="Search", width=10, pady=5, command=self.search_data).grid(row=0,
                                                                                                         column=3,
                                                                                                         padx=10,
                                                                                                         pady=10)
        showallbtn = Button(Detail_Frame, text="Show All", width=10, pady=5, command=self.fetch_data).grid(row=0,
                                                                                                           column=4,
                                                                                                           padx=10,
                                                                                                           pady=10)

        # ==========Table Frame==========
        Table_Frame = Frame(Detail_Frame, bd=4, relief=RIDGE, bg="#6550c7")
        Table_Frame.place(x=10, y=70, width=760, height=500)
        #  x = vị trí hàng y = vị trí cột
        scroll_x = Scrollbar(Table_Frame, orient=HORIZONTAL)
        scroll_y = Scrollbar(Table_Frame, orient=VERTICAL)
        self.Student_table = ttk.Treeview(Table_Frame,
                                          columns=("roll", "name", "email", "gender", "contact", "dob", "Address"),
                                          xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set)
        scroll_x.pack(side=BOTTOM, fill=X)
        scroll_y.pack(side=RIGHT, fill=Y)
        scroll_x.config(command=self.Student_table.xview)
        scroll_y.config(command=self.Student_table.yview)
        self.Student_table.heading("roll",
                                   text="Roll No.")  # ("roll","name","email","gender","contact","dob","Address")
        self.Student_table.heading("name", text="Name")
        self.Student_table.heading("email", text="Email")
        self.Student_table.heading("gender", text="Gender")
        self.Student_table.heading("contact", text="Contact")
        self.Student_table.heading("dob", text="D.O.B")
        self.Student_table.heading("Address", text="Address")
        self.Student_table['show'] = 'headings'  # xóa dòng này thừa 1 head trống
        self.Student_table.column("roll", width=100)
        self.Student_table.column("name", width=100)
        self.Student_table.column("email", width=100)
        self.Student_table.column("gender", width=100)
        self.Student_table.column("contact", width=100)
        self.Student_table.column("dob", width=100)
        self.Student_table.column("Address", width=100)

        self.Student_table.pack(fill=BOTH, expand=1)
        # them button sau click thong tin student table hien vao o nhap 
        self.Student_table.bind("<ButtonRelease-1>", self.get_cursor)
        # goi ham
        self.fetch_data()

    def on_close(self):
        self.root.destroy()
    
    def add_students(self):
        if self.Roll_No_var.get() == "" or self.name_var.get() == "":
            messagebox.showerror("Error", "All fields are required!!!")
        else:
            con = pymysql.connect(host="localhost", user="root", password="", database="students")
            cur = con.cursor()
            cur.execute("insert into data values(%s,%s,%s,%s,%s,%s,%s)", (
                self.txt_Roll.get(),
                self.txt_name.get(),
                self.txt_Email.get(),
                self.combo_gender.get(),
                self.txt_Contact.get(),
                self.txt_Dob.get(),
                self.txt_Address.get('1.0', END)
            ))
            con.commit()
            self.fetch_data()
            self.clear()
            con.close()
            messagebox.showinfo("Success", "Record has been inserted")

    # duyệt phần tử và select lên student table, show len student table
    def fetch_data(self):
        con = pymysql.connect(host="localhost", user="root", password="", database="students")
        cur = con.cursor()
        cur.execute("select * from data")
        rows = cur.fetchall()
        if len(rows) != 0: 
            self.Student_table.delete(*self.Student_table.get_children())
            for row in rows:
                self.Student_table.insert('', END, values=row)
                con.commit()

        con.close()

    # xoa toan bo cac dong khi nhap sai
    def clear(self):
        # self.Roll_No_var.set("")
        # self.name_var.set("")
        # self.email_var.set("")
        # self.gender_var.set("")
        # self.contact_var.set("")
        # self.dob_var.set("")
        # self.txt_Address.delete('1.0',END)
        self.txt_Roll.delete(0, END)
        self.txt_name.delete(0, END)
        self.txt_Email.delete(0, END)
        self.gender_var.set("")
        self.txt_Contact.delete(0, END)
        self.txt_Dob.delete(0, END)
        self.txt_Address.delete('1.0', END)

    # nhan phan tu se hien len
    def get_cursor(self, ev):
        curosor_row = self.Student_table.focus()
        contents = self.Student_table.item(curosor_row)
        row = contents['values']
        # print(row)
        self.Roll_No_var.set(row[0])
        self.name_var.set(row[1])
        self.email_var.set(row[2])
        self.gender_var.set(row[3])
        self.contact_var.set(row[4])
        self.dob_var.set(row[5])
        # khi click vao thong tin student_table address hien len dia chi cua o do va dong thoi se hien len dia chi o tiep theo 
        self.txt_Address.delete('1.0', END)
        self.txt_Address.insert(END, row[6])

    def update_data(self):
        con = pymysql.connect(host="localhost", user="root", password="", database="students")
        cur = con.cursor()
        cur.execute("update data set name=%s,email=%s,gender=%s,contact=%s,dob=%s,address=%s where roll_no=%s", (
            self.txt_name.get(),
            self.txt_Email.get(),
            self.combo_gender.get(),
            self.txt_Contact.get(),
            self.txt_Dob.get(),
            self.txt_Address.get('1.0', END),
            self.txt_Roll.get()
        ))
        con.commit()
        self.fetch_data()
        self.clear()
        con.close()

    def delete_data(self):
        con = pymysql.connect(host="localhost", user="root", password="", database="students")
        cur = con.cursor()
        # self.txt_Roll tu dong xoa khi chay file
        cur.execute("delete from data where roll_no=%s", self.Roll_No_var.get())
        con.commit()
        con.close()
        self.fetch_data()
        self.clear()
        messagebox.showinfo("Success", "Record has been deleted")

    def search_data(self):
        con = pymysql.connect(host="localhost", user="root", password="", database="students")
        cur = con.cursor()
        # note select * from data where" nen select * from data where "
        cur.execute(
            "select * from data where " + str(self.search_by.get()) + " LIKE '%" + str(self.search_txt.get()) + "%'")
        rows = cur.fetchall()
        if len(rows) != 0:
            self.Student_table.delete(*self.Student_table.get_children())
            for row in rows:
                self.Student_table.insert('', END, values=row)
            con.commit()

        con.close()

    def add_photo(self):
        conn = pymysql.connect(host="localhost", user="root", password="", database="students")
        query = "SELECT * FROM data WHERE roll_no=" + str(self.Roll_No_var.get())
        cur = conn.cursor()
        cur.execute(query)

        isRecordExist = 0  # kiểm tra xem đã có id trong database rồi cho biến này = 1 chưa = 0 để insert,update

        # duyệt từng hàng trên bản ghi nếu có tồn tại chuyển về =1
        for row in cur:
            isRecordExist = 1
        if (isRecordExist == 0):  # chưa có insert
            query = "INSERT INTO data VALUES(" + str(self.Roll_No_var.get()) + ",'" + str(self.name_var.get()) + "')"
        else:
            query = "UPDATE data SET name='" + str(self.name_var.get()) + "' WHERE roll_no=" + str(
                self.Roll_No_var.get())

        cur.execute(query)
        conn.commit()
        conn.close()

        # load thư viện
        face_cascade = cv2.CascadeClassifier("E:\Downloads\Diem_Danh_NhanDien_KhuonMat\Diem_Danh_NhanDien_KhuonMat/test/haarcascade_frontalface_default.xml")
        #face_cascade = cv2.CascadeClassifier('/test/cascades/data/haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("Unable to load cas")
        cap = cv2.VideoCapture(0)
        fontface = cv2.FONT_HERSHEY_SIMPLEX
        # # insert to db
        # id = input("Enter your ID: ")
        # name = input("Enter your Name: ")
        # insertOrUpdate(id,name)

        # index
        sampleNum = 0

        while (True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                if w <= 210 and h <=210:
                    continue
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

                if not os.path.exists('datasets/' + self.Roll_No_var.get()):
                    os.makedirs('datasets/' + self.Roll_No_var.get())

                sampleNum += 1

                cv2.imwrite(
                    'datasets/' + str(self.Roll_No_var.get()) + '/User.' + str(self.Roll_No_var.get()) + '.' + str(
                        sampleNum) + '.jpg', gray[y: y + h, x: x + w])
                cv2.putText(frame, "Collecting : " + str(sampleNum), (x + 10, y + h + 30), fontface, 1, (0, 255, 0), 2)
            cv2.imshow('Face', frame)
            cv2.waitKey(1)

            # lấy 100 ảnh
            if sampleNum > 200:
                messagebox.showinfo("Collection", "Successfully Captured!!!")
                # print("Successfully Captured")
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


root = Tk()
ob = Student(root)
root.mainloop()