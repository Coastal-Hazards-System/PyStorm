%% StormSelection.m
% By: Norberto C. Nadal-Caraballo, PhD

clc; clear; close all

%% Load Input Files

HURDATname = 'HURDATatl20230504';
%ATCFname = ['atcf',char(datetime('now','Format','yyyyMMdd'))];
ATCFname = 'atcf20240113';

load(['..\1_TC_Observations\7_ATCF_Update-Optional\out\','Index_TC_',HURDATname,'_col27_gpm_Cp_Rm_',ATCFname,'.mat'])
load(['..\1_TC_Observations\7_ATCF_Update-Optional\out\','CHS_TC_',HURDATname,'_col27_gpm_Cp_Rm_',ATCFname,'.mat'])
load(['..\2_CRLs\','CHS_Atl_CRLs_v1.6.mat'])

% Col (01) = TC #
% Col (02) = Time snap #
% Col (03) = YYYY
% Col (04) = NHC ID
% Col (05) = YYYYMM date
% Col (06) = DDHHMM date (GMT)
% Col (07) = Latitude
% Col (08) = Longitude
% Col (09) = Landfall (indicated by a 1)
% Col (10) = Hurricane (indicated by a 1)
% Col (11) = Maximum sustained wind (km/h)
% Col (12) = Minimum central pressure (hPa)
% Col (13) = Translation speed (km/h)
% Col (14) = Heading direction
% Col (15) = 63 km/h wind radii maximum extent in northeastern quadrant (km)
% Col (16) = 63 km/h wind radii maximum extent in southeastern quadrant (km)
% Col (17) = 63 km/h wind radii maximum extent in southwestern quadrant (km)
% Col (18) = 63 km/h wind radii maximum extent in northwestern quadrant (km)
% Col (19) = 93 km/h wind radii maximum extent in northeastern quadrant (km)
% Col (20) = 93 km/h wind radii maximum extent in southeastern quadrant (km)
% Col (21) = 93 km/h wind radii maximum extent in southwestern quadrant (km)
% Col (22) = 93 km/h wind radii maximum extent in northwestern quadrant (km)
% Col (23) = 119 km/h wind radii maximum extent in northeastern quadrant (km)
% Col (24) = 119 km/h wind radii maximum extent in southeastern quadrant (km)
% Col (25) = 119 km/h wind radii maximum extent in southwestern quadrant (km)
% Col (26) = 119 km/h wind radii maximum extent in northwestern quadrant (km)
% Col (27) = Radius of Maximum Winds (km)

if ~exist('.\out', 'dir')
    % Folder does not exist so create it.
    mkdir('out');
end

%% GKF Rate

K_size = 200; %unit: km
max_dist = 600; %km
maxCP = 1005; %985 hPa

%Nyrs = 2019-1938+1;

%% Load Input Files

load(['in\','NOAA_WCL_Coastline_LowRes.mat'])
NOAA_ECD1 = coastline; clear coastline;

load(['in\','NOAA_WDB_Political.mat'])
NOAA_ECD2 = political;

%% CRLs

% crl_tmp = CRL;
% crl_tmp(664:end,:)=[];
% 
% % Gulf = [1 199]; Florida = [200 327]; Atl = [328...
% apx = [25, -81];
% pt1= 267; pt2=268;
% 
% crl0 = [[80,-120];[0,-120];[0,-97.75879873];crl_tmp;[80,-61.93130721];[80,-120]];
% plot(crl0(:,2),crl0(:,1),'k')
% 
% crl1 = [[80,-120];[0,-120];[0,-97.75879873];crl_tmp(1:pt1,:);apx;[0,apx(2)];[0,-61.93130721];[80,-61.93130721];[80,-120]];
% plot(crl1(:,2),crl1(:,1),'k')
% 
% crl2 = [[80,-120];[0,-120];[0,apx(2)];apx;crl_tmp(pt2:end,:);[80,-61.93130721];[80,-120]];
% plot(crl2(:,2),crl2(:,1),'k')

%% Save Plots

folderName = ['CHS_StormSelection_Plots_',num2str(max_dist),'km'];
if ~exist(folderName, 'dir')
    % Folder does not exist so create it.
    mkdir(folderName);
end

%% Computation

CHS=CHS_TC;

%CHS(CHS(:,3)<1938,:)=[]; %Year
CHS(isnan(CHS(:,12)),:)=[]; %Central pressure
CHS(CHS(:,12)>maxCP,:)=[]; %maxCP

CHS(isnan(CHS(:,7)),:)=[]; %Lat
CHS(isnan(CHS(:,8)),:)=[]; %Lon

[~,ia,ic] = unique(CHS(:,1),'rows','stable'); %unique TCs
Nstrm = CHS(ia,1);

for N = 1:size(CRL,1) %5296 %# of nodes ~33981
    %clc; size(CRL,1)-N %#ok<NOPTS>
    clc; N %#ok<NOPTS>
    
    lat0 = CRL(N,1);
    lon0 = CRL(N,2);
    
%     if N<=265
%         crl_aux=crl1;
%     elseif N>=266
%         crl_aux=crl2;
%     end%if
    %plot(crl_aux(:,2),crl_aux(:,1),'k')
    
    k=1;
    for i = 1:size(ia,1) %# of TCs ~360 (Katrina=297; Sandy=357)
        SS_trk=[];
        SS_trk = CHS(ic(:,1)==i,:);
        
        if ~isempty(SS_trk)
%             in = inpolygon(SS_trk(:,8),SS_trk(:,7),crl_aux(:,2),crl_aux(:,1));
%             SS_trk(in,:)=[];
            
            distDeg = distance(lat0,lon0,SS_trk(:,7),SS_trk(:,8)); %output = degrees
            distKM = distdim(distDeg,'deg','km'); %converts from deg to km
            %plot(lon0,lat0,'ob',SS_trk(:,8),SS_trk(:,7),'*r')
            
            if min(distKM)<=max_dist
                
                %Distance
                id1 = find(distKM(:,1)==min(distKM(:,1)),1,'first');
                
                %Gaussian Weight
                gaussW = GaussianWeights(K_size,distKM);
                W_CP = gaussW.*(1013-SS_trk(:,12));
                iw1 = find(W_CP(:,1)==max(W_CP(:,1)),1,'first');
                %lat1(k) = SS_trk(iw1,7); lon1(k) = SS_trk(iw1,8);
                
                % Col (01) = YYYYMM date
                % Col (02) = Storm # for the year
                % Col (03) = Latitude
                % Col (04) = Longitude
                % Col (05) = Distance from CRP (km)
                % Col (06) = Forward speed (km/h)
                % Col (07) = Storm heading (deg)
                % Col (08) = Wind speed (km/h)
                % Col (09) = Cp (MINIMUM within x KM of LANDFALL)
                % Col (10) = Cp (@ LANDFALL)
                % Col (11) = Gaussian weight
                % Col (12) = Distance (km)
                % Col (13) = Radius of maximum winds (km/h) -ebtrk & GPM
                % Col (14) = Holland B (-)
                
                CHS_CRL(N).TC_Data(k,1) = SS_trk(1,5);
                CHS_CRL(N).TC_Data(k,2) = SS_trk(1,4);
                
                %Lat/Lon
                CHS_CRL(N).TC_Data(k,3) = SS_trk(iw1,7);
                CHS_CRL(N).TC_Data(k,4) = SS_trk(iw1,8);
                CHS_CRL(N).TC_Data(k,5) = distKM(iw1,1);
                
                %Translation speed
                %CHS_CRL(N).TC_Data(i,6) = SS_trk(id1,13);
                CHS_CRL(N).TC_Data(k,6) = SS_trk(iw1,13);
                
                %Heading direction (angle)
                %CHS_CRL(N).TC_Data(i,7) = SS_trk(id1,14);
                CHS_CRL(N).TC_Data(k,7) = SS_trk(iw1,14);
                
                %Wind speed
                CHS_CRL(N).TC_Data(k,8) = SS_trk(iw1,11);
                
                %Central pressure (Gauss weight)
                CHS_CRL(N).TC_Data(k,9) = SS_trk(iw1,12);
                
                %Central pressure (min distance)
                CHS_CRL(N).TC_Data(k,10) = SS_trk(id1,12);
                
                %Gaussian weight
                CHS_CRL(N).TC_Data(k,11) = gaussW(iw1,1);
                
                %Distance (km)
                CHS_CRL(N).TC_Data(k,12) = distKM(id1,1);
                
                %Radius of maximum winds (km/h) -ebtrk & GPM
                CHS_CRL(N).TC_Data(k,13) = SS_trk(iw1,27);
                
                %TC_ID: Year NHC# Name
                CHS_CRL(N).TC_ID(k,1) = Index_TC(SS_trk(iw1,1),1);
                CHS_CRL(N).TC_ID(k,2) = Index_TC(SS_trk(iw1,1),2);
                CHS_CRL(N).TC_ID(k,3) = Index_TC(SS_trk(iw1,1),3);
                
                k=k+1;
            end%if
        end%if
        
    end%for k
    
    %% Figure 1 ****************************************************************
    %     Figure1 = figure('Color',[1 1 1],'visible','off');
    %     colormap jet
    %     axes('XGrid','on','XMinorTick','on','YGrid','on','YMinorTick','on','FontSize',12);
    %     xlim([-120 0]); ylim([0 70]);
    %     set(gca,'XTick',-120:20:0)
    %     set(gca,'YTick',0:10:70)
    %     set(gca,'XTickLabel',{['120',char(176),'W'],['100',char(176),'W'],['80',char(176),'W'],...
    %         ['60',char(176),'W'],['40',char(176),'W'],['20',char(176),'W'],['0',char(176),'W']})
    %     set(gca,'YTickLabel',{['0',char(176),'N'],['10',char(176),'N'],['20',char(176),'N'],['30',char(176),'N'],...
    %         ['40',char(176),'N'],['50',char(176),'N'],['60',char(176),'N'],['70',char(176),'N']})
    %     hold on
    %
    %     plot(NOAA_ECD2(:,2),NOAA_ECD2(:,1),'LineWidth',1,'Color',[0.5 0.5 0.5]);
    %     plot(NOAA_ECD1(:,2),NOAA_ECD1(:,1),'LineWidth',1,'Color','k');
    %
    %     for j = 1:k-1
    %         %plot([lon0 lon1(j)],[lat0 lat1(j)],'LineWidth',1,'Color','b');
    %         plot(lon1(j),lat1(j),'Marker','o','MarkerSize',3,'MarkerEdgeColor','b')
    %     end%for j
    %     plot(lon0,lat0,'Marker','o','MarkerSize',5,'MarkerEdgeColor','k','MarkerFaceColor','r')
    %
    %     title({['StormSim JPM ',char(8212),' CRL ',num2str(N)]},'FontSize',14)
    %
    %     xlabel({'Longitude'},'FontSize',14);
    %     ylabel({'Latitude'},'FontSize',14);
    %
    %     %legend('NACCS','SA Phase I','SA Phase II (GoM)','Coastal Texas','SouthEast');
    %     hold off
    %     %save figure
    %     SV1 = 'CHS_CRL_';
    %     SV2 = num2str(N);
    %     %SV2 = 'All';
    %     figura1 = [folderName1,'/',strcat(SV1,SV2)];
    %     saveas(Figure1,figura1,'png')
    
    %% Figure 1 ****************************************************************
    Figure1 = figure('Color',[1 1 1],'visible','off');
    colormap jet
    axes('XGrid','on','XMinorTick','on','YGrid','on','YMinorTick','on','FontSize',12);
    xlim([-110 -50]); ylim([10 50]);
    set(gca,'XTick',-110:10:50)
    set(gca,'YTick',10:5:50)
    set(gca,'XTickLabel',{['110',char(176),'W'],['100',char(176),'W'],['90',char(176),'W'],...
        ['80',char(176),'W'],['70',char(176),'W'],['60',char(176),'W'],['50',char(176),'W']})
    set(gca,'YTickLabel',{['10',char(176),'N'],['15',char(176),'N'],['20',char(176),'N'],['25',char(176),'N'],...
        ['30',char(176),'N'],['35',char(176),'N'],['40',char(176),'N'],['45',char(176),'N'],['50',char(176),'N']})
    hold on
    
    plot(NOAA_ECD2(:,2),NOAA_ECD2(:,1),'LineWidth',1,'Color',[0.5 0.5 0.5]);
    plot(NOAA_ECD1(:,2),NOAA_ECD1(:,1),'LineWidth',1,'Color','k');
    
%     for j = 1:k-1
%         
%         peak = 1013-CHS_CRL(N).TC_Data(j,9);
%         if peak<28
%             pColor = 'g';
%         elseif peak>=28 && peak<48
%             pColor = 'y';
%         elseif peak >=48
%             pColor = 'r';
%         end%if
%         
%         %plot([lon0 lon1(j)],[lat0 lat1(j)],'LineWidth',1,'Color','b');
%         plot(lon1(j),lat1(j),'Marker','o','MarkerSize',3,'MarkerEdgeColor','k','MarkerFaceColor',pColor)
%     end%for j

    lat = CHS_CRL(N).TC_Data(:,3);
    lon = CHS_CRL(N).TC_Data(:,4);
    dp = 1013-CHS_CRL(N).TC_Data(:,9);
    
    aux1 = (dp<28); aux2 = (dp>=28 & dp<48); aux3 = dp>=48;
    h1 = plot(lon(aux1),lat(aux1),'Marker','o','MarkerSize',3,'MarkerEdgeColor','k','MarkerFaceColor','g','LineStyle','none');
    h2 = plot(lon(aux2),lat(aux2),'Marker','o','MarkerSize',3,'MarkerEdgeColor','k','MarkerFaceColor','y','LineStyle','none');
    h3 = plot(lon(aux3),lat(aux3),'Marker','o','MarkerSize',3,'MarkerEdgeColor','k','MarkerFaceColor','r','LineStyle','none');
    
    plot(lon0,lat0,'Marker','o','MarkerSize',5,'MarkerEdgeColor','k','MarkerFaceColor','b')
    
    title({['CHS ',char(8212),' Atlantic CRL ',num2str(N,'%04d')]},'FontSize',14)
    
    xlabel({'Longitude'},'FontSize',14);
    ylabel({'Latitude'},'FontSize',14);
    legend([h3 h2 h1],{'High','Med','Low'},...
                    'Location','NorthWest','FontSize',8);
    lgd = legend;
    lgd.Title.String = 'TC Intensity';
    %legend('NACCS','SA Phase I','SA Phase II (GoM)','Coastal Texas','SouthEast');
    hold off
    %save figure
    SV1 = 'CHS_Atlantic_CRL_';
    SV2 = num2str(N,'%04d');
    %SV2 = 'All';
    figura1 = [folderName,'/',strcat(SV1,SV2)];
    saveas(Figure1,figura1,'png')
    
end%for N

%% Save TC_Data Files

save(['out\','CHS_CRL_TC_HURDATatl_',num2str(max_dist),'km.mat'],'CHS_CRL');

%% END
