#!/usr/bin/env python
# coding: utf-8

# In[ ]:


n=int(input("번호:"))
k=int(input("국어:"))
e=int(input("영어:"))
m=int(input("수학:"))
p=int(input("물리:"))
sum=k+e+m+p
avg=sum/4
if 80<=avg<=100:
    r='A'
elif 60<=avg<80:
    r='B'
elif 40<=avg<60:
    r='C'
elif 20<=avg<40:
    r='D'
elif 0<=avg<20:
    r='F'
print("번호   국어   영어   수학   물리   총합   평균   학점")
print(n,"   ",k,"   ",e,"   ",m,"   ",p,"   ",sum,"  ",avg,"  ",r)

