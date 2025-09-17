function [D] = trend_D(k,I,D, epsilon_novo)
% Funkcija za biranje intervala u kome VFF ima nagle promene
n_1 = k - round(I/4)+1;
n_2 = k + round(I/4);
prag = 80;

% Deifnisanje lokalnih ekstremuma
alfa_min = min(D(n_1:n_2, 1));
alfa_max = max(D(n_1:n_2, 1));
delta_alfa = alfa_max - alfa_min;
if (delta_alfa <= prag)
    % Ako nastaju nagle promene u lokalu tada se ne dira D
    D(k,1) = L(k-I+1, k+I, epsilon_novo) - L(k-I+1,k,epsilon_novo) - L(k+1,k+I,epsilon_novo);
elseif (delta_alfa > prag)
    % Ako ne nastaju nagle promene u lokalu signal je priblizno stacionaran
    D(k,1) = alfa_min;
end
end