#!/usr/bin/env stackless2.7
# -*- coding: utf-8 -*-

from PIL import Image
import sys
import math
from datetime import datetime

filename = "edge_img_000"
image = Image.open(filename + ".jpg")

pix = image.load()
size = image.size

print pix[0, 0]
print image.size
print image

#адаптивное пороговое преобразование
min_val = 120
def trans(pix, i, j):
    res = \
        pix[i - 1, j - 1] + pix[i - 1, j] + pix[i - 1, j + 1] + \
        pix[i, j - 1] + 8 * pix[i, j] + pix[i, j + 1] + \
        pix[i + 1, j - 1] + pix[i + 1, j] + pix[i + 1, j + 1]
    res /= 16
    if res > min_val:
        return res
    else:
        return 0

image1 = Image.new(image.mode, size)
pix1 = image1.load()

for i in xrange(1, size[0] - 1):
    for j in xrange(1, size[1] - 1):
        pix1[i, j] = trans(pix, i, j)

image1.save(filename + "_trans.jpg")

#Кластеризация
size2 = size#(100, 100)

class Claster(object):
    def __init__(self, pixels):
        self.pixels = pixels
        self.x = x
        self.y = y
        self.value = sum(map(lambda pixel: pixel[2], pixels))

        self.metrics = {}

    @staticmethod
    def metrika(a, b):
        #return a.value * b.value / ((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
        return 1.0 / ((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def add_claster(self, claster):
        self.pixels.extend(claster.pixels)
        self.x = (self.x * self.value + claster.x * claster.value) / (self.value + claster.value)
        self.y = (self.y * self.value + claster.y * claster.value) / (self.value + claster.value)
        self.value += claster.value

    def __repr__(self):
        return "Claster(%s, %s, %s)" % (self.x, self.y, self.value)


clasters = []

#создание кластеров
RANGE = 10
dt = datetime.now()

for i in xrange(size2[0] / RANGE):
    for j in xrange(size2[1] / RANGE):
        pixels = []
        for m in xrange(RANGE):
            for n in xrange(RANGE):
                x = i * RANGE + m
                y = j * RANGE + n
                if pix1[x, y] > 0:
                    pixels.append((x, y, pix1[x, y]))

        if pixels:
            clasters.append(Claster(pixels))

#первоначальный расчет метрик
for i in xrange(len(clasters) - 1):
    for j in xrange(i + 1, len(clasters)):
        F = Claster.metrika(clasters[i], clasters[j])
        clasters[i].metrics[clasters[j]] = F
        clasters[j].metrics[clasters[i]] = F

#итоговое кол-во кластеров
cl_count = int(1 + 1.33 * math.log(len(clasters)))
print "count: %s" % cl_count

while len(clasters) > cl_count:
    #поиск макс метрики
    max_F = 0
    cl1 = None
    cl2 = None
    for i in xrange(len(clasters) - 1):
        for j in xrange(i + 1, len(clasters)):
            F = clasters[i].metrics[clasters[j]]
            #F = Claster.metrika(clasters[i], clasters[j])
            if F > max_F:
                max_F = F
                cl1 = clasters[i]
                cl2 = clasters[j]

    #Удаление метрик из кэша
    for claster in clasters:
        if not claster in [cl1, cl2]:
            del claster.metrics[cl1]
            del claster.metrics[cl2]

    cl1.metrics = {}

    #Объединение кластеров
    cl1.add_claster(cl2)
    clasters.remove(cl2)

    #Пересчет метрик
    for claster in clasters:
        if claster != cl1:
            F = Claster.metrika(claster, cl1)
            cl1.metrics[claster] = F
            claster.metrics[cl1] = F

    print len(clasters)

print datetime.now() - dt

#сохранение кластеризации
colors = [
    (255, 218, 185),
    (240, 255, 255),
    (112, 128, 144),
    (0, 191, 255),
    (0, 100, 0),
    (124, 252, 0),
    (205, 92, 92),
    (210, 105, 30),
    (255, 20, 147),
    (255, 250, 250),
]

image2 = Image.new("RGB", size2)
pix2 = image2.load()
for i in xrange(cl_count):
    claster = clasters[i]
    for x, y, value in claster.pixels:
        pix2[x, y] = colors[i]

image2.save(filename + "_cl.jpg")
