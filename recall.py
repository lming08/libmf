#! /usr/bin/env python
#coding:utf-8
count = 0
number = 0
real = 0
inference = 0
test = open("../ua.test")
pred = open("output")
line = test.readline()
while line:
    number+=1
    value = float(line.strip().split(" ")[2])
    compare = float(pred.readline().strip())
    if value>=2.1:
        real+=1
    if compare>=2.1:
        inference+=1
    if value >=2.1 and compare>=2.1:
        count+=1
    line = test.readline()
print "Real:",real,"\t预测购买>2.1:",inference
print "测试集:",number,"\t同时命中>2.1:",count
