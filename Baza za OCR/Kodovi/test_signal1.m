%                       Test signal 1 
%==========================================================================

function [duzina, t, p, teta, y] = test_signal1()
%definisanje parametara AR modela signala

% duzina signala
duzina = 1000;
% frekvencija signala
f(1,1:800) = 0.2;
f(1,801:duzina) = 0.4;    
% vremenski interval
t = 1 : duzina;                                                                
arg(1,t) = 2*pi*f(1,t);
% red AR modela
p = 2;   
% prvi parametar AR modela
teta_1(1,t) = -2*cos(arg(1,t));      
% drugi parametar AR modela
teta_2 = 1; 
% spajamo oba parametra (teta_1 i teta_2) u jednu matricu teta
teta = zeros(p,duzina);             
for i = 1 : duzina
    teta(:,i)=[teta_1(1,i); teta_2];                                        
end

% definisemo pocetne uslove AR modela (y(k-1) i y(k-2))
y(1,1:p) = cos(arg(1,1:p));

% definisem AR model signala preko parametara AR modela
for i = (p+1) : duzina
    y(1,i)=-teta_1(1,i) * y(1,i-1) - teta_2 * y(1,i-2);
end

% transportujemo matricu AR modela
y = y';
end