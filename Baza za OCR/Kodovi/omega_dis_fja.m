function [omega_pom_2] = omega_dis_fja(k,epsilon_novo, d, y_novo, Z_novo, teta_hat)
% racunanje omege za diskriminacionu funkciju
% preko Huberove nelinearnosti
arg = epsilon_novo(k,1) / d;
psi = min(abs(arg),k) * sign(arg);
if (y_novo(k,1) ~= Z_novo(:,k)' * teta_hat(:,k))
    omega_pom_2 = psi/arg;
else 
    omega_pom_2 = 1;
end 
end