import re
from tkinter import *
import json
from tkinter import ttk
from tkinter import messagebox
from tkinter.messagebox import *


class VerticalScrolledFrame(ttk.Frame):

    def __init__(self, parent, **kw):
        ttk.Frame.__init__(self, parent, **kw)

        scrollbar = ttk.Scrollbar(self, orient=VERTICAL)
        scrollbar.pack(fill=Y, side=RIGHT, expand=FALSE)
        canvas = Canvas(self, bd=0, highlightthickness=0,
                        yscrollcommand=scrollbar.set)
        canvas.pack(side=LEFT, fill=BOTH, expand=TRUE)
        scrollbar.config(command=canvas.yview)

        canvas.xview_moveto(0)
        canvas.yview_moveto(0)

        self.interior = interior = ttk.Frame(canvas)
        interior_id = canvas.create_window(0, 0, window=interior,
                                           anchor=NW)

        def _configure_interior(event):
            # Update the scrollbars to match the size of the inner frame.
            size = (interior.winfo_reqwidth(), interior.winfo_reqheight())
            canvas.config(scrollregion="0 0 %s %s" % size)
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the canvas's width to fit the inner frame.
                canvas.config(width=interior.winfo_reqwidth())

        interior.bind('<Configure>', _configure_interior)

        def _configure_canvas(event):
            if interior.winfo_reqwidth() != canvas.winfo_width():
                # Update the inner frame's width to fill the canvas.
                canvas.itemconfigure(interior_id, width=canvas.winfo_width())

        canvas.bind('<Configure>', _configure_canvas)


class MyCollectApp(Toplevel):  # 重点
    def __init__(self, max_len):
        super().__init__()  # 重点
        self.max_len = max_len
        self.start_id = IntVar(value=0)
        self.end_id = IntVar(value=0)
        self.title('标注范围选择')
        self.setup_ui()

    def setup_ui(self):
        row1 = Frame(self)
        row1.pack(fill="x")
        Entry(row1, textvariable=self.start_id).pack(side=LEFT)
        l1 = Label(row1, text="-", height=2, width=10)
        l1.pack(side=LEFT)
        Entry(row1, textvariable=self.end_id).pack(side=RIGHT)
        row2 = Frame(self)
        row2.pack(fill="x")
        Button(row2, text="点击确认", command=self.on_click).pack(side=RIGHT)

    def on_click(self):
        start_id = self.start_id.get()
        end_id = self.end_id.get()
        if start_id >= end_id:
            messagebox.showwarning(title='系统提示', message='开始下标大于结束下标!')
            return False
        elif end_id > self.max_len:
            messagebox.showwarning(title='系统提示', message='结束下标超出总长度!')
            return False
        self.quit()
        self.destroy()

    def get_start_end(self):
        return self.start_id.get(), self.end_id.get()


class View(Frame):

    def __init__(self, master=None):
        super(View, self).__init__(master)
        # 初始配置Optional

        self.file_dataset_path = "./data/train/all.json"
        self.save_path = "./data/save.json"
        self.page = IntVar(value=0)
        self.page_num = 0
        self.dataset = []
        self.result = []
        self.current_id = None

        """ 显示句子和实体  """
        self.frame_title = Frame(root, bd=1, relief="sunken")
        self.frame_title.pack(side="top", fill="x", ipadx=10,
                              ipady=80,
                              expand=0)
        """ 底部的功能区 """
        self.frame_bottom = Frame(root, width=100, bd=1, relief="sunken")
        self.frame_bottom.pack(side="top", fill="both", ipadx=10,
                               ipady=10,
                               expand=True)
        self.entry02 = Entry(self.frame_bottom, textvariable=self.page, width=15)
        self.page_to_button = Button(self.frame_bottom, text="page to", font=("  Times New roman", 12),
                                     command=self.page_to)
        self.entry02.pack(side='left')
        self.page_to_button.pack(side='left')

        self.insert_button()
        self.get_dataset_by_file()
        self.max_len = len(self.dataset)
        self.set_sentence()

    def insert_button(self):
        model_frame = Frame(self.frame_bottom)
        model_frame.pack(side="bottom")
        # 底部功能栏的按钮
        before_button = Button(model_frame, text="<", font=("  Times New roman", 12), command=self.before)
        entry_button = Button(model_frame, text="保存数据", font=("  Times New roman", 12), command=self.save_data)
        entry_button_1 = Button(model_frame, text="删除", font=("  Times New roman", 12), command=self.delete_data)
        entry_button_2 = Button(model_frame, text="保存文件", font=("  Times New roman", 12), command=self.save_file)
        entry_button_3 = Button(model_frame, text="保存训练文件", font=("  Times New roman", 12),
                                command=self.save_finally_file)
        next_button = Button(model_frame, text=">", font=("  Times New roman", 12), command=self.next)
        pass_button = Button(model_frame, text="pass", font=("  Times New roman", 12), compound=self.pass_data)
        # 按钮定位
        before_button.grid(row=0, column=4, padx=10, pady=10)
        entry_button.grid(row=0, column=6, padx=10, pady=10)
        entry_button_2.grid(row=0, column=14, padx=10, pady=10)
        entry_button_1.grid(row=0, column=10, padx=10, pady=10)
        entry_button_3.grid(row=0, column=12, padx=10, pady=10)
        next_button.grid(row=0, column=8, padx=10, pady=10)

    def before(self):
        if self.page_num > 0:
            self.page_num -= 1
            self.page.set(self.page_num)
            self.update_frame()

    def save_data(self):
        if self.dataset[self.page_num] in self.result:
            messagebox.showinfo("警告!", "请勿重复保存同一数据!")
        else:
            self.result.append(self.dataset[self.page_num])

    def save_file(self):
        with open(self.save_path, "w", encoding='utf-8') as file:
            text = json.dumps(self.result)
            file.write(text)
        messagebox.showinfo("提示", "保存文件成功!")

    def delete_data(self):
        for i, one in enumerate(self.result):
            if one['id'] == self.page_num:
                self.result.pop(i)
                messagebox.showinfo("提示!", "删除成功!")
                return

    def next(self):
        if self.page_num < self.max_len:
            self.page_num += 1
            self.page.set(self.page_num)
            self.update_frame()

    def pass_data(self):

        pass

    def get_dataset_by_file(self):
        with open(self.file_dataset_path) as file:
            text = file.read()
            if text == '':
                self.dataset = []
            else:
                self.dataset = json.loads(text)
        with open(self.save_path) as file:
            text = file.read()
            if text == '':
                self.result = []
            else:
                self.result = json.loads(text)

    def get_sentence(self):
        current_id = self.page_num
        if current_id < self.max_len:
            sentence = re.split("<e1>|</e1>|<e2>|</e2>", " ".join(self.dataset[current_id]['sentence']))
            return sentence, self.dataset[current_id]['relation']
        return None

    def set_sentence(self):
        sentence, relation = self.get_sentence()
        if sentence is None:
            return

        self.list_frame = VerticalScrolledFrame(self.frame_title)
        self.list_frame.pack(side="top", fill="y", ipadx=10, ipady=10,
                             expand=5)
        label_1 = Label(self.list_frame.interior, text='句子:', font=(" Times New roman", 10))
        label_1.grid(row=0, column=0)
        label_2 = Label(self.list_frame.interior, text='关系:', font=(" Times New roman", 10))
        label_2.grid(row=1, column=0)
        sentence_ = Text(self.list_frame.interior, wrap=WORD, font=("  Times New roman", 12), width=50,
                         height=3 if (len(' '.join(sentence)) / 30) < 3 else len(' '.join(sentence)) / 30)
        sentence_.tag_config("tag", background="yellow", foreground="red")
        for i, s in enumerate(sentence):
            if i == 1 or i == 3:
                sentence_.insert(INSERT, '<e1>' + s + '</e1>' if i == 1 else '<e2>' + s + '</e2>', "tag")
            else:
                sentence_.insert(INSERT, s)

        sentence_.grid(row=0, column=4, padx=2, pady=5)
        relation_ = Text(self.list_frame.interior, font=("  Times New roman", 12), width=50, height=3)
        relation_.insert(INSERT, relation)
        relation_.grid(row=1, column=4, padx=2, pady=5)

    def update_frame(self):
        self.list_frame.destroy()
        self.set_sentence()

    def page_to(self):
        num = int(self.page.get())
        if num < self.max_len:
            self.page_num = num
            self.update_frame()
        else:
            messagebox.showinfo("警告!", "页面超出范围!")

    def event_handler(self, event):
        if event.keysym == 'Left':
            self.before()
        elif event.keysym == 'Right':
            self.next()
        elif event.keysym == "1":
            self.save_data()
        elif event.keysym == "2":
            self.delete_data()
        elif event.keysym == "3":
            self.save_finally_file()
        elif event.keysym == "4":
            self.save_file()

    def save_finally_file(self):
        mx5 = askokcancel(title='确认/取消对话框', message='是否全部标注完成?')
        if mx5:
            app = MyCollectApp(self.max_len)
            app.mainloop()
            start_id, end_id = app.get_start_end()
            app.destroy()
            print(start_id, end_id)
            for page in range(start_id, end_id):
                if self.dataset[page] in self.result:
                    continue
                else:
                    self.dataset[page]['relation'] = "Other"
                    self.result.append(self.dataset[page])
            self.result.sort(key=lambda x: x['id'])
            self.save_file()
        else:
            return


if __name__ == "__main__":
    root = Tk()
    root.geometry("800x500")
    root.title("关系标注")
    index = View(master=root)
    root.bind_all('<KeyPress>', index.event_handler)
    root.mainloop()
