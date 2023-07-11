
from tkinter import *

root = Tk()

root.title("窗口测试")

def eventhandler(event):
    if event.keysym == 'Left':
        print('按下了方向键左键')
    elif event.keysym == 'Right':
        print('按下了方向键右键！')


root.bind_all('<KeyPr1ess>', eventhandler)


root.mainloop()