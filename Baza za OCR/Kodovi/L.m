%           Logaritam funkcije verodostojnosti
%==============================================================
function [rez] = L(a,b,epsilon_novo)

suma_eps = 0;
for i = a : b
    suma_eps = suma_eps + epsilon_novo(i,1)^2;
end
rez = (b-a+1)* log((1/(b-a+1))*suma_eps);
end