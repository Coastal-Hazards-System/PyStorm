function [Hd_out,Hd_mean,Hd_stdv] = headingZeroDegree_adj(Hd_in,type,Hd_mean)

% Type 1 = adjust DSRR (Parametric distribution)
% Type 2 = adjust Hd individual observations
% Type 3 = adjust Hd individual observations with pre-computed 'Hd_mean'

switch type
    case 1
        %hding = -170:10:180; old
        hding = -179:1:180;
        Hd_mean = sum( hding.*Hd_in,'omitnan' );
        Hd_stdv = sqrt( sum(Hd_in.*(hding-Hd_mean).^2,'omitnan') );
        Hd_aux = hding - Hd_mean;
        Hd_aux(Hd_aux>180) = Hd_aux(Hd_aux>180) - 360; %correct angles > 180
        Hd_aux(Hd_aux<-180) = Hd_aux(Hd_aux<-180) + 360; %correct angles < -180
        %scatter(Hd_aux,Hd_in)
        Hd_out = interp1(Hd_aux,Hd_in,hding,'linear','extrap');
        Hd_out(Hd_out<0)=0; %correct for possible negative values after 'extrap'
        Hd_out = Hd_out/sum(Hd_out); %normalize in case sum < 1.0
    case 2
        Hd_mean = mean(Hd_in,"omitnan");
        Hd_out = Hd_in - Hd_mean;
        Hd_out(Hd_out>180) = Hd_out(Hd_out>180) - 360; %correct angles > 180
        Hd_out(Hd_out<-180) = Hd_out(Hd_out<-180) + 360; %correct angles < -180
    case 3
        Hd_out = Hd_in - Hd_mean;
        Hd_out(Hd_out>180) = Hd_out(Hd_out>180) - 360; %correct angles > 180
        Hd_out(Hd_out<-180) = Hd_out(Hd_out<-180) + 360; %correct angles < -180
end

% %% "Distance-Weighted" Mean
% param_mu_dw = sum( param.*wght,'omitnan' ) / sum(wght,'omitnan');
% %% "Distance-Weighted" Standard Deviations
% param_sigma_dw = sqrt( sum(wght.*(param-param_mu_dw).^2,'omitnan') ./ ((N-1)/N*sum(wght,'omitnan')) );