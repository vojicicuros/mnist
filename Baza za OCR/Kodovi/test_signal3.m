%% Generisem test signal 2 i parametre tog signala
function [duz, t, p, teta, y] = test_signal3()
duz = 1000;
t = 1 : duz;                %vremenski interval
p = 8;

% teta_1(1,1:200) = 0.9900;
% teta_1(1,201:duz) = 0.1879;
% teta_2 = +0.7497;
% teta_3(1,1:200) = +1.0588;
% teta_3(1,201:duz) = +0.1574;
% teta_4 = +0.3656;
% teta_5 = 0.5748;
% teta_6(1,1:200) = 0.8412;
% teta_6(1,201:duz) = 0.0795;
% teta_7 = +0.1119;
% teta_8 = +0.1209;

teta_1(1,1:200) = 0.1618;
teta_1(1,201:duz) = 1.1046;
teta_2 = +0.1194;
teta_3 = -0.4714;
teta_4 = -0.2502;
teta_5 = 0.2646;
teta_6 = 0.7059;
teta_7 = +0.1839;
teta_8 = +0.0032;


%Spajam sve parametre u jednu matricu
teta = zeros(p,duz);
for i = 1 : duz
    teta(:,i) = [teta_1(1,i); teta_2; teta_3; teta_4;...
        teta_5; teta_6; teta_7; teta_8];
end
% teta = teta * 1.5;

%definisem pocetne uslove AR modela (y(k-1) i y(k-2))
y(1,1:p) = teta(1:p,1);

%definisem AR model signala preko parametara AR modela
for i = (p+1) : duz
    y(1,i) = -teta_1(1,i)*y(1,i-1) - teta_2 * y(1,i-2)...
        - teta_3 * y(1,i-3) - teta_4 * y(1,i-4) - teta_5 * y(1,i-5)...
        - teta_6 * y(1,i-6) - teta_7 * y(1,i-7) - teta_8 * y(1,i-8);
end

y = y';
end