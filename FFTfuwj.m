clc
clear
close all
%采样频率1600hz
fs=1600;
%data = csvread('192126.csv',1,0);
%data = csvread('2025101511195226.csv',1,0);
%data = csvread('2025101511270629.csv',1,0);
data = csvread('2025101511272529.csv',1,0);
data(:,1)=data(:,1)/fs;
data(:,2)=data(:,2)-mean(data(:,2));
data(:,3)=data(:,3)-mean(data(:,3));
data(:,4)=data(:,4)-mean(data(:,4));
%%Z积分运算得到速度
vz(1)=0;
len=length(data(:,1));
for i=2:len
vz(i)=trapz(data(1:i,1),data(1:i,4)/100);
end

figure
subplot(4,1,1);
plot(data(:,1),data(:,2))
%title('192126.csv');
%title('2025101511195226.csv');
%title('2025101511270629.csv');
title('2025101511272529.csv');
ylabel('x(gals)')
subplot(4,1,2);
plot(data(:,1),data(:,3))
ylabel('y(gals)')
subplot(4,1,3);
plot(data(:,1),data(:,4))
ylabel('z(gals)')
xlabel('t(s)')
subplot(4,1,4);
plot(data(:,1),vz)
ylabel('v(m/s)')
xlabel('t(s)')

%选取fft的时间端，本次第10s开始%
t=10;
%选择点数1S 1024点，2秒2048点，4秒4096 8秒8192，16秒16384点
n=8192;
%取样开始点
nxyz1=fs*10;
%取样结束点
nxyz2=fs*10+n-1;
px=fft(data(nxyz1:nxyz2,2));
py=fft(data(nxyz1:nxyz2,3));
pz=fft(data(nxyz1:nxyz2,4));
%双侧频谱
px2=abs(px/n);
py2=abs(py/n);
pz2=abs(pz/n);
%单侧频谱
px1=px2(1:(n/2));
px1(2:end-1)=2*px1(2:end-1);
py1=py2(1:(n/2));
py1(2:end-1)=2*py1(2:end-1);
pz1=pz2(1:(n/2));
pz1(2:end-1)=2*pz1(2:end-1);
f=fs*(1:(n/2))/n;
figure
subplot(4,1,1);
plot(f,px1);
%title('192126.csv.csv');
%title('2025101511195226.csv');
%title('2025101511270629.csv');
title('2025101511272529.csv')
ylabel('x(gals)')
subplot(4,1,2);
plot(f,py1);
ylabel('y(gals)')
subplot(4,1,3);
plot(f,pz1);
ylabel('z(gals)')
xlabel('HZ')
subplot(4,1,4);
plot(data(:,1),vz)
ylabel('v(m/s)')
xlabel('t(s)')


