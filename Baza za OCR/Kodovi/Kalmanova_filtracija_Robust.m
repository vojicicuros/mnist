function [p,Z,y,y_sum, y_hat, teta, epsilon, t, duzina, omega, beli_sum] = Kalmanova_filtracija_Robust(izbor, odnos)

%generisanje test signala u zavisnosti od izbora
if  izbor == 1
    [duzina,t,p,teta,y] = test_signal1(); 
elseif izbor == 2
    [duzina,t,p,teta,y] = test_signal2();
elseif izbor == 3
    [duzina,t,p,teta,y] = test_signal3();
end

% Racunanje suma koji utice na signal
[beli_sum, varijansa_suma] = sum_signala(duzina, odnos);
% Sklairanje suma u odnosu na signal da SNR bude 10dB
[beli_sum, faktor] = skaliranje(beli_sum, y, duzina);
% Dodavanje belog Gausovskog suma na signal
y_sum = y + beli_sum;

% Definisanje Z izvedeno iz AR modela
Z = zeros(p,duzina); %definisanje unapred zbog brzine alokacije memorije
for i = (p+1) : duzina
    for j = 1 : p
        Z(j,i) = -y_sum(i-j,1);
    end
end

G=zeros(p,1);
% Definisemo matricu G
G(1,1)=1;
% Sum AR modela 
u(1,t) = beli_sum;
% Ulazni sum
v = (G*u)';
% Varijansa merenog suma
sigma_u= varijansa_suma;

% Definisemo matricu F
F = zeros(p,p,duzina);
for i = 1 : duzina
    for j = 1 : p        
        prva_vrsta(i,j) = -teta(j,i);
    end
    dodajem = horzcat(eye(p-1), zeros((p-1),1));
    F(:,:,i) = vertcat(prva_vrsta(i,:), dodajem); 
end
% Kovarijansa merenog suma
Q = sigma_u^2*G*(G');
R = 0.08^2;
% Definisemo matricu H
H = zeros(1,p);
H(1,1) = 1;
%  greska predikcije (rezidual merenja)
epsilon = zeros(duzina,1);
for i = 1 : duzina
        epsilon(i,1) = y_sum(i,1) - Z(:,i)' * teta(:,i);
end

% Pocetna vrednost ulaza modela stanja
xo_hat = ones (p,1);
x_hat(:,:,1) = xo_hat;

% Pocetna vrednost kovarijanse greske estimacije
P = zeros(p,p,1);

omega = zeros(duzina,1);
x_line = zeros(p,1,duzina);
M = zeros(p,p,duzina);
s = zeros(duzina,1);
K= zeros(p,1,duzina);
y_hat = zeros(duzina,1);
y1 = y_sum;
%  greska predikcije (rezidual merenja)
epsilon2 = zeros(duzina,1);
for k = 2: duzina
    % 1. korak predikcije
    % predikcija stanja sistema
    x_line(:,:,k) = F(:,:,k-1)*x_hat(:,:,k-1);
    % matrica kovarijanse greske predikcije
    M(:,:,k) = F(:,:,k-1)*P(:,:,k-1)*(F(:,:,k-1)')+Q;
    % predikcija izlaza sistema
    y_hat(k,1) = H * x_line(:,:,k);
    % 2. korak estimacije 
    % matrica kovarijanse reziduala
    s(k,1) = sqrt(H*M(:,:,k)*(H')+R);
    % izracunavanje reziduala
    epsilon2(k,:) = y1(k,:) - y_hat(k,1);
    % tezinska forma
    l=k;
  if round(epsilon2(k,:)*10000)/10000== 0 
    omega_pom = 1;
  else  
    arg = round((epsilon2(k,:) / s(k,1))*10000)/10000;
    psi = min(abs(arg),l) * sign(arg);
    omega_pom = round((psi / arg)*10000)/10000;
  end
    omega(k,1) = omega_pom;
    
    % matrica kalmanovog pojacanja
    K(:,:,k) = omega(k,1)*(round(M(:,:,k)*(H')*10000)*(s(k,1)^(-1)))/10000;
    
    % matrica kovarijanse greske estimacije
    P(:,:,k) = (eye(p)-K(:,:,k)*H)*M(:,:,k);
    % procena stanja
    x_hat(:,:,k) = x_line(:,:,k)+K(:,:,k)*epsilon2(k,1);
    
end
end