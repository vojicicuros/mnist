%                            Skaliranje
%==========================================================================
% Skaliranje amplitude gausovog belog suma

function [beli_sum, faktor] = skaliranje(beli_sum, y, duzina)

max_amp_sig = max(y);
max_amp_sum = max(beli_sum);
dozv_max_amp_sum = max_amp_sig / (sqrt(10));

% ako je maximalna amplituda veca od dozvoljene
if max_amp_sum > dozv_max_amp_sum
    faktor = dozv_max_amp_sum / max_amp_sum;
    for i = 1 : duzina
        %umanjivanje svakog odbirka sume za vrednost izracunatog faktora
        beli_sum(i,1) = beli_sum(i,1) * faktor;
    end

% ako je maksimalna amplituda suma manja od dozvoljene
elseif max_amp_sum < dozv_max_amp_sum
    faktor = max_amp_sum / dozv_max_amp_sum;
    for i = 1 : duzina
        %uvecavanje svakog odbirka suma za vredsnost izracunatog faktora
        beli_sum(i,1) = beli_sum(i,1) * faktor;
    end
end

end