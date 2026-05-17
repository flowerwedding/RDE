function Chaos_p=GenerateChaos(max_it)
% [~,dim]=size(X);
Chaos_p=zeros(14,max_it);
Chaos=zeros(1,max_it);

for ChaosSystemName_p=1:14

if ChaosSystemName_p==1%Logisticmap
    z1=0.152;
    a=4.0;%a=3.6 3.7 3.8 3.9
    Chaos(1)=z1;
    for j=1:max_it
        Chaos(j+1)=a*Chaos(j)-Chaos(j)*a*Chaos(j);
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==2%PWLCMmap
    z1=0.152;
    Chaos(1)=z1;
    a=0.7;
    for j=1:max_it
        if Chaos(j)<a &&Chaos(j)>0
           Chaos(j+1)=Chaos(j)/a;
        elseif Chaos(j)<1 &&Chaos(j)>=a
           Chaos(j+1)=(1-Chaos(j))/(1-a); 
        end
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==3%Singermap
    z1=0.002;
    a=1.073;%0.9-1.08
    Chaos(1)=z1;
    for j=1:max_it
        Chaos(j+1)=a*(7.86*Chaos(j)-23.31*Chaos(j)^2+28.75*Chaos(j)^3-13.302875*Chaos(j)^4);
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==4%Sinemap
    z1=0.152;
    Chaos(1)=z1;
    for j=1:max_it
        Chaos(j+1)=sin(pi*Chaos(j));
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==5%Gaussmap;
    z1=0.152;%%%%%%%0.84,0.85,0.86，0.3，0.4，0.5，0.6d都很不好，大部分是0，初始解很重要。
              %%%但是0.344，0.3456得到的解还不错，0.34就不可以。其他的值得到的解差不很多，
    Chaos(1)=z1;
    for j=1:max_it
    if Chaos(j)==0
       Chaos(j+1)=0;
    else
       val=1/Chaos(j);  
       Chaos(j+1)=val-floor(val);
    end
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==6%Tentmap
    z1=0.152;
    a=0.4;
    Chaos(1)=z1;
    for j=1:max_it
        if Chaos(j)<=a&&Chaos(j)>0
           Chaos(j+1)=Chaos(j)/a;
        elseif Chaos(j)>a&&Chaos(j)<=1
               Chaos(j+1)=(1-Chaos(j))/(1-a); 
        end
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==7%Bernoullimap
    z1=0.152;
    a=0.4;%%%%%此处将a的取值改为0.4，除了0.5结果都不错。0.5的话结果不久就陷入0。
    Chaos(1)=z1;
    for j=1:max_it
        if Chaos(j)<=(1-a)&&Chaos(j)>0
    Chaos(j+1)=Chaos(j)/(1-a);
        elseif Chaos(j)>(1-a)&&Chaos(j)<1
            Chaos(j+1)=(Chaos(j)-(1-a))/a; 
        end
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==8%Chebyshevmap
    z1=0.152;
    a=5;
    Chaos(1)=z1;
    for j=1:max_it
     Chaos(j+1)=cos(a*cos(Chaos(j))^(-1));
     if Chaos(j+1)<0
         Chaos(j+1)=abs(Chaos(j+1));
     else
         Chaos(j+1)=Chaos(j+1);
     end
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==9%Circlemap
    z1=0.152;
    b=0.5;
    a=2.2;%%%%根据别的文献，改变了a的值，还可以。原来为0.2，结果很不好。
    Chaos(1)=z1;
    for j=1:max_it
     val=(a/(2.0*pi))*sin(2*pi*Chaos(j));
	 Chaos(j+1)=Chaos(j)+b-(val-floor(val));
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==10%Cubicmap
    z1=0.242;
    a=2.59;%%%%
    Chaos(1)=z1;
    for j=1:max_it
    Chaos(j+1)=a*Chaos(j)*(1-Chaos(j)^2);
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==11%sinusoidalmap
    z1=0.74;%初始值不同，得到的图相差很大，特别是在小于0.44时
    a=2.3;
    Chaos(1)=z1;
    for j=1:max_it
        Chaos(j+1)=a*(Chaos(j)^2)*sin(pi*Chaos(j));  
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

if ChaosSystemName_p==12%ICMICmap
    z1=0.152;
    Chaos(1)=z1;
    a=70;%%%a=1，4，11，14，时图形是极端的，其他值时正常，随着a的增大图发生的变化，当a趋向无穷大时，图变化很小很小。
    for j=1:max_it
       Chaos(j+1)=sin(a/Chaos(j));
        if Chaos(j+1)<0
         Chaos(j+1)=abs(Chaos(j+1));
        else
            Chaos(j+1)=Chaos(j+1);
        end
    end
Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
end

% if ChaosSystemName_p==13;%Iterativemap
%     z1=0.152;
%     a=0.7;
%     Chaos(1)=z1;
%     for j=1:max_it;
%         Chaos(j+1)=sin(a*pi/Chaos(j));
%         if Chaos(j+1)<0
%             Chaos(j+1)=abs(Chaos(j+1));
%         end
%     end
% Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
% end
% 
% if ChaosSystemName_p==14;%Intermittency Map
%     z1=0.152;
%     Chaos(1)=z1;
%     for j=1:max_it;
%         if Chaos(j)<=0.7 && Chaos(j)>0
%             Chaos(j+1)=0.0001+Chaos(j)+(0.2999*(Chaos(j))^2/0.49);
%         elseif Chaos(j)>0.7 && Chaos(j)<1
%             Chaos(j+1)=(Chaos(j)-0.7)/0.3;
%         end
%     end
%     Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
% end
% 
% if ChaosSystemName_p==15;%Zaslavsky Map
%     Chaos(1)=0.1;
%     y(1)=0.1;
%     e=0.3;
%     r=5;
%     omega=100;
%     k=9;
%     a=1.885;
%     Chaos(1)=z1;
%     for i=2:max_it
%         Chaos(i)=mod(Chaos(i-1)+omega/(2*pi)+(a*omega)/(2*pi*r)*(1-exp(-r))*y(i-1)+(k/r)*(1-exp(-r))*cos(2*pi*Chaos(i-1)),1);
%         y(i)=exp(-r)*(y(i-1)+e*cos(2*pi*Chaos(i-1)));
%     end
%     Chaos_p(ChaosSystemName_p,:)=Chaos(2:max_it+1);
% end


if ChaosSystemName_p==13%uniform distribution
    Chaos_p(ChaosSystemName_p,:)=rand(1,max_it);
end

if ChaosSystemName_p==14%normal random
    Chaos_p(ChaosSystemName_p,:)=randn(1,max_it);
end


end


    
    
    
 

