close all;clear all;clc;
% for iii = 1 : 10    % ovaj red ukljuciti u kod kada su prisutne sve slike a
                      % donji red komentarisati
for iii = 1:10
    
    % unos slike koja se obradjuje
    naziv_slike = ['BROJ_' int2str(iii-1) '.jpg'];
    slika = imread(naziv_slike);
    
    figure(1);
    imshow(slika);
    
    
    % rotiranje slike radi preglednosti
    slika=imrotate(slika, -90);
    
    figure(2);
    imshow(slika);
    [n,m,l] = size(slika);
    
    % uklanjanje ivica, ako slucajno nije lepo skenirana slika
    slika = slika(100:n-100,100:m-100,:);  
    [n,m,l] = size(slika);
    
    % pretvaranje u binarnu sliku radi procesiranja
    level = graythresh(slika);             
    binarna_slika = im2bw(slika,level);     
    
    figure(3);
    imshow(binarna_slika);
    

    %% izvlacenje matrice iz skeniranog A4 lista
    
    suma_po_vrstama = [];
    s = 0;
    
    for i = 1 : n
        for j = 1 : m
            s = s + binarna_slika(i,j);
        end;
        suma_po_vrstama(i) = s;
        s=0;
    end;
    
    i=1;
    while suma_po_vrstama(i) > m-5
        i=i+1;
    end
    gornjagranica = i;
    
    i = n;
    while suma_po_vrstama(i) > m-5
        i=i-1;
    end
    donjagranica = i;
    
    suma_po_kolonama = [];
    s = 0;
    for i = 1 : m
        for j = 1 : n
            s = s + binarna_slika(j,i);
        end;
        suma_po_kolonama(i) = s;
        s=0;
    end
    
    i = 1;
    while suma_po_kolonama(i) > n-5
        i = i + 1;
    end
    levagranica = i;
    
    i = m;
    while suma_po_kolonama(i) > n-5
        i = i - 1;
    end
    desnagranica = i;
    
    slika = slika(gornjagranica:donjagranica,levagranica:desnagranica,:);
    
    figure(4)
    imshow(slika);
    
    %% uklanjanje linija u matrici
    
    pomocna = slika (11,11,2);
    [n,m,l] = size(slika);
    
    br_za_vrste = floor(m/12);

    for i = 1 : 13
        if i == 1
            levagranica = 1;
            desnagranica = 15;
            slika(:,levagranica:desnagranica,:) = pomocna;
            levagranica = -15;
        elseif i == 13
            levagranica = levagranica + br_za_vrste;
            desnagranica = m;
            slika(:,levagranica:desnagranica,:) = pomocna;
        else
            levagranica = levagranica + br_za_vrste;
            desnagranica = desnagranica + br_za_vrste;
            slika(:,levagranica:desnagranica,:) = pomocna;
        end            
    end

    
    br_za_kolone = floor(n/10);
   
    for i = 1 : 11
        if i == 1
            gornjagranica = 1;
            donjagranica = 15;
            slika(gornjagranica:donjagranica,:,:) = pomocna;
            gornjagranica = -15;
        elseif i == 11
             gornjagranica = gornjagranica + br_za_vrste;
             donjagranica = n;
             slika(gornjagranica:donjagranica,:,:) = pomocna;
        else
            gornjagranica = gornjagranica + br_za_kolone;
            donjagranica = donjagranica + br_za_kolone;
            slika(gornjagranica:donjagranica,:,:) = pomocna;
        end
    end
    
    figure(5)
    imshow(slika);
    
    naziv_slike_za_cuvanje=['cifra_' int2str(iii-1) '.jpg'];
    imwrite(slika,naziv_slike_za_cuvanje);
    
    level = graythresh(slika);
    binarna_slika = im2bw(slika,level);
    
    figure(6);
    imshow(binarna_slika);
    close all;clear all;clc;     % ukljuciti red u kod kada su prisutne
                                   % slike svih cifara
end
