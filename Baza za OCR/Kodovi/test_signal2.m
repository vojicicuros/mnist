%                       Test signal 2
%==========================================================================

function [duzina, t, p, teta, y] = test_signal2()
%definisanje parametara AR modela signala

p = 2;                            %red AR modela
duzina = 2000;                    %duzina signala
t = 1 : duzina;                   %vremenski interval

%promena frekvencije
f = zeros(1,duzina);
f(1,1:500) = 0.1;
for i= 500 : 800
    f(1,i) = f(1,i-1)+0.001;
end
f(1,801:1050) = 0.4;
f(1,1051:1250) = 0.1;
for i= 1250 : 1400
    f(1,i) = f(1,i-1)+0.002;
end
f(1,1401:1600) = 0.4;
f(1,1601:1800) = 0.1;
f(1,1801:2000) = 0.25;

arg(1,t) = 2*pi*f(1,t);

%definisanje parametara teta za test signal 2
teta1 = zeros(1,duzina);
for i = 1 : duzina
    teta11(1,i) = -2*cos(arg(1,i));
end
teta2 = 1;

%Spajam oba parametra (teta1 i teta2) u jednu matricu
teta = zeros(p,duzina);
for i = 1 : duzina
    teta(:,i) = [teta1(1,i); teta2];
end

%definisanje pocetnih uslova AR modela (y(k-1) i y(k-2))
y(1,1:p) = cos(arg(1,1:p));

%definisanje AR modela signala preko parametara AR modela
for i = (p+1) : duzina
    y(1,i) = -teta1(1,i)*y(1,i-1) - teta2 * y(1,i-2) ;
end

y = y';
end