from task.train_teacher import *
from task.train_student import *
import tensorflow as tf

ckpt='./save_model/teacher/teacher1'

train_tracher(ckpt)
train_student(ckpt)