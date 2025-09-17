function primer11

% program ucitava skenirane cifre 2 i 8
% vrsi njihovu binarizaciju i kropovanje
% sracuvana normirani intenzitet beline u gornjoj i donjoj polovini slike

N2=112 % broj uzoraka cifre 2
for ii=1:N2
    if ii<10
        ime=['test200' num2str(ii)];
    else
        if ii<100
            ime=['test20' num2str(ii)];
        else
            ime=['test2' num2str(ii)];
        end
    end
    c2=imread(ime);
    c2b=255*sign(c2-240);
    [m,n]=size(c2b);
    i=1;while min(c2b(i,:))==255;i=i+1;end
    j=m;while min(c2b(j,:))==255; j=j-1;end
    k=1;while min(c2b(:,k))==255; k=k+1;end
    l=n;while min(c2b(:,l))==255; l=l-1;end
    c2bc=c2b(i:j,k:l);
    [m,n]=size(c2bc);c2bcn=c2bc/255;
    x1=sum(sum(c2bcn(1:round(m/2),:)));
    x1=x1/((round(m/2)*n));
    x2=sum(sum(c2bcn(round(m/2)+1:m,:)));
    x2=x2/((m-round(m/2))*n);
    x3=sum(sum(c2bcn(:,1:round(n/2))));
    x3=x3/((round(n/2)*m));
    x4=sum(sum(c2bcn(:,round(n/2)+1:n)));
    x4=x4/((n-round(n/2))*m);
    X2(ii,1:4)=[x1 x2 x3 x4];
end


N8=112 % broj uzoraka cifre 8
for ii=1:N8
    if ii<10
        ime=['test800' num2str(ii)];
    else
        if ii<100
            ime=['test80' num2str(ii)];
        else
            ime=['test8' num2str(ii)];
        end
    end
    c2=imread(ime);
    c2b=255*sign(c2-240);
    [m,n]=size(c2b);
    i=1;while min(c2b(i,:))==255;i=i+1;end
    j=m;while min(c2b(j,:))==255; j=j-1;end
    k=1;while min(c2b(:,k))==255; k=k+1;end
    l=n;while min(c2b(:,l))==255; l=l-1;end
    c2bc=c2b(i:j,k:l);
    [m,n]=size(c2bc);c2bcn=c2bc/255;
    x1=sum(sum(c2bcn(1:round(m/2),:)));
    x1=x1/((round(m/2)*n));
    x2=sum(sum(c2bcn(round(m/2)+1:m,:)));
    x2=x2/((m-round(m/2))*n);
    x3=sum(sum(c2bcn(:,1:round(n/2))));
    x3=x3/((round(n/2)*m));
    x4=sum(sum(c2bcn(:,round(n/2)+1:n)));
    x4=x4/((n-round(n/2))*m);
    X8(ii,1:4)=[x1 x2 x3 x4];
end

figure(1);plot(X2(:,1),X2(:,2),'*',X8(:,1),X8(:,2),'o');
legend('cifre 2', 'cifre 8');xlabel('x1');ylabel('x2');
figure(2);plot(X2(:,3),X2(:,4),'*',X8(:,3),X8(:,4),'o')
figure(3);plot(X2(:,1),X2(:,3),'*',X8(:,1),X8(:,3),'o')




keyboard;