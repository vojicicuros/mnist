%                            Sum signala
%==========================================================================

function [beli_sum, varijansa_suma] = sum_signala(duz, odnos)
% definisanje kontaminiranog, belog Gausovskog suma
sr_vr_kont_suma = 0.1;
sigma_0=sqrt(0.3);
var_kont_suma = sigma_0;
kont_beli_sum = sr_vr_kont_suma + var_kont_suma .* randn(duz,1);

% definisanje nominalnog, belog Gausovskog suma
sr_vr_nom_suma = 0;
sigma_1 = sigma_0/sqrt(10);
var_nom_suma = sigma_1;
nom_beli_sum = sr_vr_nom_suma + var_nom_suma .* (2*(rand(duz,1)-0.5));

% kombinovanje ova dva suma u zavisnosti od izabranog odnosa
beli_sum = zeros(duz,1);
for i = 1 : duz
    br = rand(1);
    if br <= (1-odnos)
        beli_sum(i,1) = nom_beli_sum(i,1);
    else
        beli_sum(i,1) = kont_beli_sum(i,1);
    end
end

varijansa_suma = var(beli_sum);

end