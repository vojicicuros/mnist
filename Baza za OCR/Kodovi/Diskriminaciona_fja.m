%           Diskriminaciona funkcija
%==========================================================================

function [teta_hat,z1_novo,lambda1]=Diskriminaciona_fja(p,Z,y_hat,teta,epsilon2,duzina,omega,C1)

% Diskriminaciona funkcija - pocetni uslovi i potrebne promenljive

% Velicina prozora diskriminacione funkcije
I=50; 
% Izlazni signal nakon Kalmanove predfiltracije je sada ulazni signal
y_novo=y_hat;
% Definisanje faktora skaliranja
d=median(abs(y_novo-median(y_novo)))/0.6745; 

% Definisanje Z
Z_novo=Z;
for i=(p+1):duzina
    for j=1:p
        Z_novo(j,i)=-y_novo(i-j,1);
    end
end

% Pocetno epsilon i teta_hat
teta_hat=teta;
epsilon_novo=epsilon2;
for i=1:I
    epsilon_novo(i,1)=y_novo(i,1)-Z_novo(:,i)'*teta_hat(:,i);
end

% Definisanje konstanti za izracunavanje lambda
N_min=20; % za nestacionarne delove signala-mala duzina memorije
N_max=500;% za stacionarne delove signala-dugacka memorija
D_min=0; % min diskriminacione fje
D_max=1; % max diskriminacione fje
D=zeros(duzina,1);
lambda_min=1-1/N_min;  % min FF
lambda_max=1-1/N_max;  % max FF
omega_novo=omega; % izjednacavanje zbog pocetnih vrednosti
lambda1=zeros(duzina,1);

%% Diskriminaciona funkcija - algoritam
M=zeros(p,p,duzina);
K=zeros(p,1,duzina);
P=zeros(p,p,duzina);
for k=(1+I):(duzina-I)
% Definisanje greske predikcije (sa novim vrednostima)
epsilon_novo(k,1)=y_novo(k,1)-Z_novo(:,k)'*teta_hat(:,k-1);
% Racunanje tezinske forme (omega_novo)
[omega_pom_2]=omega_dis_fja(k,epsilon_novo,d,y_novo,Z_novo,teta_hat);
omega_novo(k,1)=omega_pom_2;
% Racunanje diskriminacione f-je preko logaritma f-je verodostojnosti
D(k,1)=L(k-I+1,k+I,epsilon_novo)-L(k-I+1,k,epsilon_novo)-L(k+1,k+I,epsilon_novo);
% Modifikacija D preko trenda diskriminacione funkcije
[D]=trend_D(k,I,D, epsilon_novo);
% Racunanje lambda preko diskriminacione funkcije (slika 1.)
lambda1(k,1)=((lambda_max-lambda_min)/(D_min-D_max))*(D(k,1)-D_max)+lambda_min;
% Racunanje novog maksimuma diskriminacione funkcije
D_max=max(D(1:k,1));
%RRLS algoritam
M(:,:,k) = P(:,:,k-1) / lambda1(k,1);                                   
    pom_1 = M(:,:,k) * Z_novo(:,k) * omega_novo(k,1);                      
    pom_2 = 1 + Z_novo(:,k)' * pom_1;                                      
    % Matrica pojacanja
    K(:,:,k) = pom_1/pom_2;                                               
    % Pomocna konstanta
    C = C1;
    % Matrica kovarijanse greske estimacije
    P(:,:,k) = C *eye(p) - K(:,:,k)* (Z_novo(:,k)' ) * M(:,:,k);           
    % Estimirana vrednost parametara
    teta_hat(:,k) = teta_hat(:,k-1) + K(:,:,k) * epsilon_novo(k,1);   
   %modeliranje signala na osnovu estimiranih parametra
    z1_novo(k,1)=Z_novo(:,k)'*teta_hat(:,k-1);
end