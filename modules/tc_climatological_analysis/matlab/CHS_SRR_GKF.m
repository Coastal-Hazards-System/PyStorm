%% HURDATv2_SRR_GKF_km.m
% By: Norberto C. Nadal-Caraballo, PhD

clc; clear; close all

%% Kernel size

K_size = 200; %unit: km
max_dist = 600; %unit km
%Nyrs = 2021-1938+1;
HURDAT_Year = 2023;
%Hurr_Start = datetime(HURDAT_Year+1,6,1);
%Hurr_End = datetime(HURDAT_Year+1,11,30);
%ATCF_Date = datetime(2022,11,11);
%ATCF_PartialYear = min(split(between(Hurr_Start,ATCF_Date,'days'),'days'),183)/183;
Nyrs = (HURDAT_Year-1938+1) %+ATCF_PartialYear;

hding = -179:1:180; %Headings (range of)

%% Load Input Files

load(['..\2_CRLs\','CHS_Atl_CRLs_v1.6.mat'])
load(['..\3_TC_Selection\out\','CHS_CRL_TC_HURDATatl_',num2str(max_dist),'km.mat'])

if ~exist('.\out', 'dir')
    % Folder does not exist so create it.
    mkdir('out');
end

%% Computation All

minDP = 8;

for j = 1:size(CRL,1)
    
    CHS_TC = CHS_CRL(j).TC_Data;
    CHS_TC(:,9)=1013-CHS_TC(:,9);
    CHS_TC(CHS_TC(:,1)<193800,:)=[];
    CHS_TC(CHS_TC(:,9)<minDP,:)=[];
    
    Out_dist_D = CHS_TC(:,12);
    
    Wi = 1/(sqrt(2*pi)*K_size) * exp(-1/2*(Out_dist_D/K_size).^2);
    SRR(j,1) = 1/Nyrs * sum(Wi,"omitnan");
    
end%for j

save(['out\','SRR_TC_All_',num2str(max_dist),'km.mat'],'SRR');

for j = 1:size(CRL,1)
    
    CHS_TC = CHS_CRL(j).TC_Data;
    CHS_TC(:,9)=1013-CHS_TC(:,9);
    CHS_TC(CHS_TC(:,1)<193800,:)=[];
    CHS_TC(CHS_TC(:,9)<minDP,:)=[];
    
    Out_dist_D = CHS_TC(:,12);
    Out_dist_Hd = CHS_TC(:,7);

    hdgr = Out_dist_Hd;
    hdiff = AzimuthDiff(hding,hdgr);
    
    Wi = 1/(sqrt(2*pi)*K_size) * exp(-1/2*(Out_dist_D'/K_size).^2);
    Li = 1/Nyrs * sum(Wi,"omitnan");
    
    Wd=[];Ld=[];
    for h = 1:size(hding,2)      
        Wd(:,h) = 1/(sqrt(2*pi)*30) * exp(-1/2*(hdiff(:,h)/30).^2);
        Ld(:,h) = 1/Nyrs * sum(Wd(:,h) .* Wi',"omitnan");
    end%for h
    
     dirSRR = Ld/Li;
     Hd_pdf0 = dirSRR/sum(dirSRR); %plot(hding,Hd_pdf0)
     [Hd_pdf,Hd_mean,Hd_stdv] = headingZeroDegree_adj(Hd_pdf0,1); %plot(hding,Hd_pdf)
     Hd_cdf = [0,cumsum(Hd_pdf)]; %plot([-180,hding],Hd_cdf)

     DSRR(j).pdf = Hd_pdf;
     DSRR(j).cdf = Hd_cdf;
     DSRR(j).mean = Hd_mean;
     DSRR(j).stdv = Hd_stdv;
    
end%for j

save(['out\','DSRR_TC_All_',num2str(max_dist),'km.mat'],'DSRR');

%% Low Intensity

minDP = 8;
maxDP = 28; 

for j = 1:size(CRL,1)
    
    CHS_TC = CHS_CRL(j).TC_Data;
    CHS_TC(:,9)=1013-CHS_TC(:,9);
    CHS_TC(CHS_TC(:,1)<193800,:)=[];
    CHS_TC(CHS_TC(:,9)<minDP | CHS_TC(:,9)>=maxDP,:)=[];
    
    Out_dist_D = CHS_TC(:,12);
    
    Wi = 1/(sqrt(2*pi)*K_size) * exp(-1/2*(Out_dist_D/K_size).^2);
    SRR(j,1) = 1/Nyrs * sum(Wi,"omitnan");
    
end%for j

save(['out\','SRR_TC_LI_',num2str(max_dist),'km.mat'],'SRR');

for j = 1:size(CRL,1)
    
    CHS_TC = CHS_CRL(j).TC_Data;
    CHS_TC(:,9)=1013-CHS_TC(:,9);
    CHS_TC(CHS_TC(:,1)<193800,:)=[];
    CHS_TC(CHS_TC(:,9)<minDP | CHS_TC(:,9)>=maxDP,:)=[];
    
    Out_dist_D = CHS_TC(:,12);
    Out_dist_Hd = CHS_TC(:,7);
    
    hdgr = Out_dist_Hd;
    hdiff = AzimuthDiff(hding,hdgr);
    
    Wi = 1/(sqrt(2*pi)*K_size) * exp(-1/2*(Out_dist_D'/K_size).^2);
    Li = 1/Nyrs * sum(Wi,"omitnan");
    
    Wd=[];Ld=[];
    for h = 1:size(hding,2)
        Wd(:,h) = 1/(sqrt(2*pi)*30) * exp(-1/2*(hdiff(:,h)/30).^2);
        Ld(:,h) = 1/Nyrs * sum(Wd(:,h) .* Wi',"omitnan");
    end%for h

    dirSRR = Ld/Li;
    Hd_pdf0 = dirSRR/sum(dirSRR); %plot(hding,Hd_pdf0)
    [Hd_pdf,Hd_mean,Hd_stdv] = headingZeroDegree_adj(Hd_pdf0,1); %plot(hding,Hd_pdf)
    Hd_cdf = [0,cumsum(Hd_pdf)]; %plot([-180,hding],Hd_cdf)

    DSRR(j).pdf = Hd_pdf;
    DSRR(j).cdf = Hd_cdf;
    DSRR(j).mean = Hd_mean;
    DSRR(j).stdv = Hd_stdv;

end%for j

save(['out\','DSRR_TC_LI_',num2str(max_dist),'km.mat'],'DSRR');

%% Medium Intensity

minDP = 28;
maxDP = 48; 

for j = 1:size(CRL,1)
    
    CHS_TC = CHS_CRL(j).TC_Data;
    CHS_TC(:,9)=1013-CHS_TC(:,9);
    CHS_TC(CHS_TC(:,1)<193800,:)=[];
    CHS_TC(CHS_TC(:,9)<minDP | CHS_TC(:,9)>=maxDP,:)=[];
    
    Out_dist_D = CHS_TC(:,12);
    
    Wi = 1/(sqrt(2*pi)*K_size) * exp(-1/2*(Out_dist_D/K_size).^2);
    SRR(j,1) = 1/Nyrs * sum(Wi,"omitnan");
    
end%for j

save(['out\','SRR_TC_MI_',num2str(max_dist),'km.mat'],'SRR');

for j = 1:size(CRL,1)
    
    CHS_TC = CHS_CRL(j).TC_Data;
    CHS_TC(:,9)=1013-CHS_TC(:,9);
    CHS_TC(CHS_TC(:,1)<193800,:)=[];
    CHS_TC(CHS_TC(:,9)<minDP | CHS_TC(:,9)>=maxDP,:)=[];
    
    Out_dist_D = CHS_TC(:,12);
    Out_dist_Hd = CHS_TC(:,7);
    
    hdgr = Out_dist_Hd;
    hdiff = AzimuthDiff(hding,hdgr);
    
    Wi = 1/(sqrt(2*pi)*K_size) * exp(-1/2*(Out_dist_D'/K_size).^2);
    Li = 1/Nyrs * sum(Wi,"omitnan");
    
    Wd=[];Ld=[];
    for h = 1:size(hding,2)
        Wd(:,h) = 1/(sqrt(2*pi)*30) * exp(-1/2*(hdiff(:,h)/30).^2);
        Ld(:,h) = 1/Nyrs * sum(Wd(:,h) .* Wi',"omitnan");
    end%for h
    
    dirSRR = Ld/Li;
    Hd_pdf0 = dirSRR/sum(dirSRR); %plot(hding,Hd_pdf0)
    [Hd_pdf,Hd_mean,Hd_stdv] = headingZeroDegree_adj(Hd_pdf0,1); %plot(hding,Hd_pdf)
    Hd_cdf = [0,cumsum(Hd_pdf)]; %plot([-180,hding],Hd_cdf)

    DSRR(j).pdf = Hd_pdf;
    DSRR(j).cdf = Hd_cdf;
    DSRR(j).mean = Hd_mean;
    DSRR(j).stdv = Hd_stdv;
    
end%for j

save(['out\','DSRR_TC_MI_',num2str(max_dist),'km.mat'],'DSRR');

%% High Intensity

minDP = 48;

for j = 1:size(CRL,1)
    
    CHS_TC = CHS_CRL(j).TC_Data;
    CHS_TC(:,9)=1013-CHS_TC(:,9);
    CHS_TC(CHS_TC(:,1)<193800,:)=[];
    CHS_TC(CHS_TC(:,9)<minDP,:)=[];
    
    Out_dist_D = CHS_TC(:,12);
    
    Wi = 1/(sqrt(2*pi)*K_size) * exp(-1/2*(Out_dist_D/K_size).^2);
    SRR(j,1) = 1/Nyrs * sum(Wi,"omitnan");
    
end%for j

save(['out\','SRR_TC_HI_',num2str(max_dist),'km.mat'],'SRR');

for j = 1:size(CRL,1)
    
    CHS_TC = CHS_CRL(j).TC_Data;
    CHS_TC(:,9)=1013-CHS_TC(:,9);
    CHS_TC(CHS_TC(:,1)<193800,:)=[];
    CHS_TC(CHS_TC(:,9)<minDP,:)=[];
    
    Out_dist_D = CHS_TC(:,12);
    Out_dist_Hd = CHS_TC(:,7);
    
    hdgr = Out_dist_Hd;
    hdiff = AzimuthDiff(hding,hdgr);
    
    Wi = 1/(sqrt(2*pi)*K_size) * exp(-1/2*(Out_dist_D'/K_size).^2);
    Li = 1/Nyrs * sum(Wi,"omitnan");
    
    Wd=[];Ld=[];
    for h = 1:size(hding,2)
        Wd(:,h) = 1/(sqrt(2*pi)*30) * exp(-1/2*(hdiff(:,h)/30).^2);
        Ld(:,h) = 1/Nyrs * sum(Wd(:,h) .* Wi',"omitnan");
    end%for h
    
    dirSRR = Ld/Li;
    Hd_pdf0 = dirSRR/sum(dirSRR); %plot(hding,Hd_pdf0)
    [Hd_pdf,Hd_mean,Hd_stdv] = headingZeroDegree_adj(Hd_pdf0,1); %plot(hding,Hd_pdf)
    Hd_cdf = [0,cumsum(Hd_pdf)]; %plot([-180,hding],Hd_cdf)

    DSRR(j).pdf = Hd_pdf;
    DSRR(j).cdf = Hd_cdf;
    DSRR(j).mean = Hd_mean;
    DSRR(j).stdv = Hd_stdv;
    
end%for j

save(['out\','DSRR_TC_HI_',num2str(max_dist),'km.mat'],'DSRR');

%% Plot

% load SRR_TC_All.mat
% SRR200_All = SRR*400;
% % for i = 4:597
% %     SRR200_All(i,1)=mean(SRR200_All(i-3:i+3,1));
% % end%for i
% % 
% load SRR_TC_LI.mat
% SRR200_LI = SRR*400;
% % for i = 4:597
% %     SRR200_LI(i,1)=mean(SRR200_LI(i-3:i+3,1));
% % end%for i
% % 
% load SRR_TC_MI.mat
% SRR200_MI = SRR*400;
% % for i = 4:597
% %     SRR200_MI(i,1)=mean(SRR200_MI(i-3:i+3,1));
% % end%for i
% % 
% load SRR_TC_HI.mat
% SRR200_HI = SRR*400;
% % for i = 4:597
% %     SRR200_HI(i,1)=mean(SRR200_HI(i-3:i+3,1));
% % end%for i
% 
% %txt = {'Galveston, TX'; 'New Orleans, LA'};
%     
% % Figure 1 ****************************************************************
% Figure1 = figure('Color',[1 1 1],'visible','on');
% colormap jet
% axes('XGrid','on','XMinorTick','on','YGrid','on','YMinorTick','on','FontSize',12);
% xlim([0 670]); ylim([-0.6 1.4]);
% % set(gca,'XTick',-120:20:0)
% set(gca,'YTick',0:0.2:1.4)
% % set(gca,'XTickLabel',{['120',char(176),'W'],['100',char(176),'W'],['80',char(176),'W'],...
% %     ['60',char(176),'W'],['40',char(176),'W'],['20',char(176),'W'],['0',char(176),'W']})
% % set(gca,'YTickLabel',{['0',char(176),'N'],['10',char(176),'N'],['20',char(176),'N'],['30',char(176),'N'],...
% %     ['40',char(176),'N'],['50',char(176),'N'],['60',char(176),'N'],['70',char(176),'N']})
% hold on
% 
% % plot(1:668,SRR200_All,'LineWidth',1,'LineStyle','--','Color','k');
% % plot(1:668,SRR200_LI,'LineWidth',1,'Color','k');
% % plot(1:668,SRR200_LI+SRR200_MI,'LineWidth',1,'Color','b');
% % plot(1:668,SRR200_LI+SRR200_MI+SRR200_HI,'LineWidth',1,'Color','r');
% 
% %plot(1:668,SRR200_All,'LineWidth',1,'LineStyle','--','Color','k');
% plot(1:668,SRR200_LI+SRR200_HI+SRR200_MI,'LineWidth',1,'Color','k');
% plot(1:668,SRR200_HI+SRR200_MI,'LineWidth',1,'Color','b');
% plot(1:668,SRR200_HI,'LineWidth',1,'Color','r');
% 
% 
% %lo = 0.01;
% lo=-0.55;
% 
% % text(10,SRR200_HI(10)+lo,'- Port Isabel, TX','rotation',90,'FontSize',8)
% % text(48,SRR200_HI(48)+lo,'- Galveston, TX','rotation',90,'FontSize',8)
% % text(99,SRR200_HI(99)+lo,'- Grand Isle, LA','rotation',90,'FontSize',8)
% % text(114,SRR200_HI(114)+lo,'- Biloxi, MS','rotation',90,'FontSize',8)
% % text(131,SRR200_HI(131)+lo,'- Pensacola, FL','rotation',90,'FontSize',8)
% % text(156,SRR200_HI(156)+lo,'- Apalachicola, FL','rotation',90,'FontSize',8)
% % text(195,SRR200_HI(195)+lo,'- Clearwater, FL','rotation',90,'FontSize',8)
% % text(213,SRR200_HI(213)+lo,'- Cape Coral, FL','rotation',90,'FontSize',8)
% % %text(231,SRR200_HI(231)+lo,'- Key West, FL','rotation',90,'FontSize',8)
% % text(245,SRR200_HI(245)+lo,'- Miami, FL','rotation',90,'FontSize',8)
% % text(282,SRR200_HI(282)+lo,'- Daytona Beach, FL','rotation',90,'FontSize',8)
% % text(324,SRR200_HI(324)+lo,'- Charleston, SC','rotation',90,'FontSize',8)
% % text(385,SRR200_HI(385)+lo,'- Duck, NC','rotation',90,'FontSize',8)
% % text(410,SRR200_HI(410)+lo,'- Ocean City, DE','rotation',90,'FontSize',8)
% % %text(423,SRR200_HI(423)+lo,'- Atlantic City, NJ','rotation',90,'FontSize',8)
% % text(463,SRR200_HI(463)+lo,'- Newport, RI','rotation',90,'FontSize',8)
% % text(490,SRR200_HI(490,'FontSize',14)+lo,'- Boston, MA','rotation',90,'FontSize',8)
% % text(538,SRR200_HI(538)+lo,'- Eastport, ME','rotation',90,'FontSize',8)
% % text(583,SRR200_HI(583)+lo,'- Nova Scotia','rotation',90,'FontSize',8)
% 
% %text(2,lo,'Port Isabel, TX','rotation',90,'FontSize',8)
% text(22,lo,'Corpus Christi, TX','rotation',90,'FontSize',8)
% text(53,lo,'Galveston, TX','rotation',90,'FontSize',8)
% text(109,lo,'Grand Isle, LA','rotation',90,'FontSize',8)
% text(129,lo,'Biloxi, MS','rotation',90,'FontSize',8)
% text(148,lo,'Pensacola, FL','rotation',90,'FontSize',8)
% text(175,lo,'Apalachicola, FL','rotation',90,'FontSize',8)
% text(219,lo,'Clearwater, FL','rotation',90,'FontSize',8)
% text(239,lo,'Cape Coral, FL','rotation',90,'FontSize',8)
% %text(231,lo,'- Key West, FL','rotation',90,'FontSize',8)
% text(274,lo,'Miami, FL','rotation',90,'FontSize',8)
% text(314,lo,'Daytona Beach, FL','rotation',90,'FontSize',8)
% text(327,lo,'Jacksonville, FL','rotation',90,'FontSize',8)
% text(347,lo,'Savannah, GA','rotation',90,'FontSize',8)
% text(361,lo,'Charleston, SC','rotation',90,'FontSize',8)
% %text(430,lo,'Duck, NC','rotation',90,'FontSize',8)
% text(391,lo,'Wilmington, NC','rotation',90,'FontSize',8)
% text(438,lo,'Norfolk, VA','rotation',90,'FontSize',8)
% text(471,lo,'Atlantic City, NJ','rotation',90,'FontSize',8)
% text(489,lo,'Long Beach, NY','rotation',90,'FontSize',8)
% text(519,lo,'Newport, RI','rotation',90,'FontSize',8)
% text(545,lo,'Boston, MA','rotation',90,'FontSize',8)
% text(668,lo,'Eastport, ME','rotation',90,'FontSize',8)
% %text(583,lo,'Nova Scotia','rotation',90,'FontSize',8)
% 
% 
% %plot(X_vec1(j),Y_vec1(j),'Marker','o','MarkerSize',6,'MarkerEdgeColor','k','MarkerFaceColor','r')
% 
% %title({['StormSim-PCHA',char(8212),' Storm Recurrence Rate']},'FontSize',14)
% title({['StormSim-PCHA ',char(8212),' Atlantic CONUS']},'FontSize',14)
% 
% xlabel({'CHS Coastal Reference Location (CRL)'},'FontSize',14);
% ylabel({'SRR_2_0_0_k_m (TC/yr)'},'FontSize',14);
% 
% legend(['\Deltap \geq   8 hPa'],['\Deltap \geq 28 hPa'],['\Deltap \geq 48 hPa'],'Location','NorthWest');
% set(legend,'Location','NorthWest','FontSize',10);
% hold off
% %save figure
% SV1 = 'SRR_';
% SV2 = 'CRLs_78_1938-2018';
% %SV2 = 'All';
% figura1 = [strcat(SV1,SV2)];
% saveas(Figure1,figura1,'png')



%% END
