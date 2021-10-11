%%% classlabel -1 progressor; -0 nonprogressor
% nucleiseg_nucleiclusterfinding_extractfullfeature_OroCavity_v2(4,5)
% JHU data /mnt/projects/CSE_BME_AXM788/data/JHU_Oropharyngeal/JHU_48_ROIs_Tiles/
% v2 includes new input arguments to specify the path/save path and image
% image_format='.mat' or '.png'
% suffix
% extract nuclei from tile image from mat files
% v3 specify if extract CCM features
% v4 use epistroma mask files to guid the feature extraction, i.e., we only
% extract features from where we have the epistroma mask
% v5 extract more features based on the TMA data modeling. Basically the
% CRL and fractal features. July 7, 2016
% v6 fixes the bugs due to no epi/stroma in the mask
% v7 add BS/BSinCCG features
% v8 add hosoya features
% v9 add parameter that can switch to do TMA spot image of tile from
% WSI(which require reading the location information)
% v10 add parameter that can control the nuclei seg parameter i.e., p.scales=[para_nulei_scale_low:2:para_nulei_scale_high];
% also include the FeDeG features
% v11 add "corrected" LoCoM,COrE features (the original one is not completely incorrect, but just another way to compute them), and name them as "cLoCoM/cCCM" and "cCOrE".
% V12 add robustness for the file name conversion
% V13 opt in the nuclei clustering and nuclei pre-segmentation map
% V14 clean the un-neccesary file after creating the feature vector for
% each image
% V15 consider the tile that only has epi or stroma, and we put all zeros
% in either feature vector, put just one line of text on the feature
% description
% v15_get_im_on_the_fly deal with the case that we do not have image patches, but
% know the xy location information
% strNucleiMaskPath and strEpiStrMaskPath should contain files name as the same as the image,
% v16 inherit from v15_get_im_on_the_fly, update the FeDeG features
% calculateion. cat features inside the function
% v17 add nuclei wise haralick feature from Haojia

% and include Tumor Microenvironment Orientation Patterns
% (TuMop) features


% NOTE: the assumption is that

%%% this function is from nucleiseg_nucleiclusterfinding_extractfullfeature
function Lextractfeature_v17(idxBegin,idxEnd,strWSIPath,flag_epistroma,strPathIM,strSavePath,strEpiStrMaskPath,strNucleiMaskPath,flag_have_nulei_mask,flag_nuclei_cluster_finding,image_format,flag_Hosoya,flag_haralick,flag_CCM,flag_WSI,para_nulei_scale_low,para_nulei_scale_high,flagRandomPatch)
%% parameter initialization
% addpath(genpath('../Code/George_GraphFeatureExtraction'));
% addpath(genpath('/Users/chenglu/Nutstore/PathImAnalysis_Program/Program/Features/George_GraphFeatureExtraction'));
% addpath('C:\Nutstore\Nutstore\Repository\nuclei_segmentation_code\Fast_Single_Pass_Voting');
% addpath(genpath('/Users/chenglu/Nutstore/PathImAnalysis_Program/Program/OropharyngealSCC_TMA'));
% if ispc()
%     addpath(genpath('F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\George_GraphFeatureExtraction'));
%     addpath(genpath('F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\FractalAnalysis\boxcount'));
%
%     addpath('F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\OralCavity');
%     addpath('F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\feature_driven_subgraphs_FeDeG');
%     addpath('F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\AppearenceAdjust\staining_normalization');
% end
% addpath(genpath('/Users/chenglu/Nutstore/PathImAnalysis_Program/Program/OralCavity'));
% idxBegin=554;idxEnd=554;

%% turn num
if ~ispc()
    idxBegin=str2num(idxBegin);
    idxEnd=str2num(idxEnd);
    flag_epistroma=str2num(flag_epistroma);
    flag_haralick=str2num(flag_haralick);
    flag_CCM=str2num(flag_CCM);
    flag_Hosoya=str2num(flag_Hosoya);
    flag_have_nulei_mask=str2num(flag_have_nulei_mask);
    flag_nuclei_cluster_finding=str2num(flag_nuclei_cluster_finding);
    flag_WSI=str2num(flag_WSI);
    para_nulei_scale_low=str2num(para_nulei_scale_low);
    para_nulei_scale_high=str2num(para_nulei_scale_high);
end

% if nargin<10
%     flag_WSI=1;
%     para_nulei_scale_low=6;
%     para_nulei_scale_high=16;
% else
%     if ~ispc()
%         flag_WSI=str2num(flag_WSI);
%         para_nulei_scale_low=str2num(para_nulei_scale_low);
%         para_nulei_scale_high=str2num(para_nulei_scale_high);
%     end
% end
%% begin to work

% Modified by Chuheng 11-09-2020
idxUse = idxBegin:idxEnd;
if flag_epistroma
    %     dirMask=dir([strEpiStrMaskPath '*_mask.png']);
    dirMask=dir([strEpiStrMaskPath '*.png']);
    dirIM=dir([char(strPathIM) '*' image_format]);
    if flagRandomPatch(1)
        idxUse = randperm(length(dirMask),min(length(dirMask),flagRandomPatch(2)));
    else
        idxUse = 1:length(dirMask);
    end
else
    dirIM=dir([char(strPathIM) '*' image_format]);
    if flagRandomPatch(1)
        idxUse = randperm(length(dirIM),min(length(dirIM),flagRandomPatch(2)));
    else
        idxUse = 1:length(dirIM);
    end
end
Num_level4Haralick_nuclei_wise=8;



model = load('F:\CWRU-Research\Feature Extract Code\SpaTIL_v2\example_data\lymp_svm_matlab_40x.mat');
mdl = model.model;
if flag_have_nulei_mask
    dirNucMsk = dir(strNucleiMaskPath);
    dirNucMsk = dirNucMsk(3:end);
    nucMskStr = string({dirNucMsk.name})';
    nucMskStr = extractBefore(nucMskStr,'.png');
    ImStr = string({dirIM.name})';
    ImStr = extractBefore(ImStr,'.png');
    [~,idxUse,~] = intersect(ImStr,nucMskStr);
end
se = strel('disk', 10);
load NewImgDescription.mat
% Modification of Chuheng Ends here.
for jjj = 1:length(idxUse)
    i = idxUse(jjj);
     try
        
        allFeats=[];alldescription=[];
        if flag_epistroma
            curMask=dirMask(i).name;
            %         tempidx=strfind(curMask,'.png');
            curIM=curMask(1:end-11);
            curMask = curMask(1:end-4);
            fprintf('The mask idx is %d, the mask name is %s, image ID is %s\n',i,curMask, curIM);
        else
            curIM=dirIM(i).name;
            tempidx=strfind(curIM,image_format);
            curIM=curIM(1:tempidx(end)-1);
            fprintf('The mask idx is %d, no ES mask so far, image ID is %s\n',i, curIM);
        end
        
        if strcmp(image_format,'.mat')
            curName=[strPathIM curIM '.mat'];
            if exist(curName,'file')==2
                load([curName]);
            else
                curTile=imread([strPathIM curIM '.png']);%show(curTile);
                if flag_WSI
                    strlocation=sprintf('%s%s_location_in_ROI.mat',strPathIM,curIM);
                    if exist(strlocation,'file')==2
                        load(strlocation);
                    end
                end
                image_format='.png';
            end
        else
            if flag_WSI
                curName=[curIM image_format];
                tmp=strsplit(curIM,'_x');
                curWSIname=tmp{1};
                tmp=strsplit(tmp{end},'_y');
                tile_x=str2num(tmp{1});tile_y=str2num(tmp{2});
                %         ff=imfinfo([strWSIPath curWSIname '.tiff']);
                %         curTile=imread([strWSIPath curWSIname '.tiff'],'PixelRegion',{[tile_y tile_y+2048-1],[tile_x tile_x+2048-1]});%show(curTile);
                curTile=imread([strWSIPath curWSIname '.tiff'],'PixelRegion',{[tile_x tile_x+2048-1],[tile_y tile_y+2048-1]});%show(curTile);
                
                %         curTile=imread([strPathIM curIM image_format]);%show(curTile);
                %         temp=curName;
                %         temp=temp(1:end-4);
                
                strlocation=sprintf('%s%s_location_in_ROI.mat',strPathIM,curIM);
                if exist(strlocation,'file')==2
                    load(strlocation);
                end
            else
                curName=[curIM image_format];
                curTile=imread([strPathIM curName]);
            end
        end
        
        if flag_WSI
            idxtile=strfind(curName,'Tile');
            strtile=curName(idxtile:end);
            %         idxTile=sscanf(strtile,'%d');
            idxd=regexp(strtile,'\d'); idxdd=strfind(strtile,'.');
            idxTile=str2num(strtile(idxd(1):idxdd));
        end
        %a = randomError;
        
        %% nuclei segmentation for current tile
        I=curTile; clear curTile; %show(I);
        
        temp=strsplit(curIM,'_x');
        curPID=temp{1};
        LcreateFolder(sprintf('%s%s',strSavePath,[curPID '/']));
        curName=[curIM image_format];
        
        strAllFeatsMat=sprintf('%s%s%s_allFeats.mat',strSavePath,[curPID '/'],[]);
        strAllFeatsMat_name=sprintf('%s%sallFeatsNames.mat',strSavePath,[curPID '/']);
        if exist(strAllFeatsMat,'file')~=2
            strSaveCur=sprintf('%s%s%s_nuclei.mat',strSavePath,[curPID '/'],[]);
            if exist(strSaveCur,'file')~=2
                if flag_have_nulei_mask
                    bwNuclei=logical(imread([strNucleiMaskPath strrep(curName,'.png','.png')]));%show(bwNuclei)%LshowBWonIM(bwNuclei,I);
                    if length(size(bwNuclei))>2
                        bwNuclei = bwNuclei(:,:,1);
                    end
                    real_bounds = Lmask2bounds(logical(bwNuclei));
                    nuclei=Lbounds2nuclei(real_bounds);
                    
                    L = bwlabel(logical(bwNuclei));%show(L)
                    [I_norm, ~, ~] = normalizeStaining(I);
                    %% Check TIL detection
                    %real_bounds = Lmask2bounds(logical(bwNuclei));
                    %% Remove TIL
                    cc_nuc = bwconncomp(bwNuclei);
                    [~,nucFeatures] = getNucLocalFeatures(I_norm,L);
                    [lympho_label,~,~] = predict(mdl,nucFeatures(:,1:7));%  If label == 1, cell is a TIL
                    idx = find(lympho_label ~= 1);
                    bwNuclei = ismember(labelmatrix(cc_nuc),idx);
                    %% Check TIL detection
                    %                     show(I); show(I_norm);show(L)
                    %                     L2 = bwlabel(logical(bwNuclei));
                    %
                    %                     nuclei=Lbounds2nuclei(real_bounds);
                    %
                    %                     show(L2)
                    %                     show(I_norm);hold on
                    %                     for k = 1:length(nuclei)
                    %                         if lympho_label(k)==1
                    %                             plot(nuclei{k}(:,2), nuclei{k}(:,1), 'y-.', 'LineWidth', 1);
                    %                         else
                    %                             plot(nuclei{k}(:,2), nuclei{k}(:,1), 'r-', 'LineWidth', 1);
                    %                         end
                    %                     end
                    %% Continue previous work
                    
                    L = bwlabel(logical(bwNuclei));
                    real_bounds = Lmask2bounds(logical(bwNuclei));
                    nuclei=Lbounds2nuclei(real_bounds);
                    I_norm=I_norm(:,:,1);
                    properties = LNuclear_regionProperties(I_norm, L);
                    
                    save(strSaveCur,'nuclei','properties');
                    % save(strSaveCur,'properties');
                    disp('nuclei result saved; ');
                else
                    % turn black into white to reduce the normolization effect
                    bbw=I(:,:,1)<2;%show(bbw);
                    r=I(:,:,1);g=I(:,:,2);b=I(:,:,3);
                    r(bbw)=255;g(bbw)=255;b(bbw)=255;
                    I=cat(3,r,g,b);%show(I);
                    [I_norm, ~, ~] = normalizeStaining(I);
                    %         clear I;%show(I(:,:,2));
                    I_normRed=I_norm(:,:,1);%show(I_normRed);show(I_norm(:,:,3));
                    %         p.scales=[2:2:16];
                    p.scales=[para_nulei_scale_low:2:para_nulei_scale_high];
                    try
                        disp('begin nuclei segmentation;');
                        %                         [nuclei, properties] = nucleiSegmentationV2(I_normRed,p);
                        [nuclei, properties] = nucleiSegmentationV2(I_normRed,p); %show(I(:,:,1));
                        %%%% check
                        %                 show(I_normRed);hold on;
                        %                 for k = 1:length(nuclei)
                        %                     if nuclei_label(k)
                        %                         plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 1);
                        %                     else
                        %                         plot(nuclei{k}(:,2), nuclei{k}(:,1), 'b-', 'LineWidth', 1);
                        %                     end
                        %                 end
                        %                 hold off;            %%% remove the false positive region
                        [nuclei, properties]=LremoveFalsePositive_from_contour_v2(nuclei,properties,I);
                        %             imwrite(I,'1.png');
                        %             tt=imread('1_mask.png');show(tt); bwtt=tt>5;
                        %             LshowBWonIM(bw_nuclei,I,1); %LshowBWonIM(bwtt,I);
                        %             saveas(gcf,sprintf('%s%s_nuclei.png',strSavePath,curName));
                    catch ME
                        idSegLast = regexp(ME.identifier, '(?<=:)\w+$', 'match');
                        %clean up for possible error conditions here. Rethrow if
                        %             unknown error.
                        switch idSegLast{1}
                            case 'nomem'
                                disp('Out of Memory!');
                            case 'couldNotReadFile'
                                disp('file not there!');
                            otherwise
                                %An unexpected error happened
                                rethrow(ME)
                        end
                        continue
                    end
                    %%% save results
                    if flag_WSI
                        save(strSaveCur,'nuclei','properties','curRow', 'curCol','idxTile');
                        disp('nuclei result saved; ');
                    else
                        save(strSaveCur,'nuclei','properties');
                        disp('nuclei result saved; ');
                    end
                end
            else
                disp('nuclei file found, loading it');
                load(strSaveCur);
            end
            % check nuclei seg result
            %                         show(I);hold on;
            %                     for k = 1:length(nuclei)
            %                             plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 1);
            %                     end
            %                     hold off;            %%% remove the false positive region
            
            %%  !!!!!!!! below code are for the feature extractions without epi/stroma seperation !!!!!!!!
            %% nuclei cluster finding - meanshift
            if flag_nuclei_cluster_finding
                bandWidth=30;
                %             strSaveCur=sprintf('%s%s%s_nuclei.mat',strSavePath,[curPID '/'],curName);
                
                str=sprintf('%s%sMSCellCluster_BW%d_IM%s.mat',strSavePath,[curPID '/'],bandWidth,curName);
                if exist(str,'file')~=2
                    try
                        %% use meanshift to form cell clusters, use location info only
                        AlldataPts=[];
                        tempC=[properties.Centroid];
                        AlldataPts(1:length(properties),1)=tempC(1:2:end);
                        AlldataPts(1:length(properties),2)=tempC(2:2:end);
                        
                        dataPts=AlldataPts(:,1:2);
                        disp('begin cell cluster finding;');
                        [clustCent,data2cluster,cluster2dataCell] = MeanShiftCluster(dataPts',bandWidth);
                        
                        %         bandWidth60=60;
                        %         [clustCent60,data2cluster60,cluster2dataCell60] = MeanShiftCluster(dataPts',bandWidth60);
                    catch ME
                        idSegLast = regexp(ME.identifier, '(?<=:)\w+$', 'match');
                        %clean up for possible error conditions here. Rethrow if
                        %             unknown error.
                        switch idSegLast{1}
                            case 'nomem'
                                disp('Out of Memory!');
                            case 'couldNotReadFile'
                                disp('file not there!');
                            otherwise
                                %An unexpected error happened
                                rethrow(ME)
                        end
                        continue
                    end
                    save(str,'clustCent','data2cluster','cluster2dataCell','bandWidth');
                    %
                    %     str=sprintf('%sPro_MSCellCluster_BW%d_%d_IM%d.mat',strPathIM,bandWidth,bandWidth60,i);
                    %     save(str,'clustCent','data2cluster','cluster2dataCell','bandWidth',...
                    %         'clustCent60','data2cluster60','cluster2dataCell60','bandWidth60');
                    disp('nuclei cluster result saved; ');
                else
                    disp('nuclei cluster file found, loading it');
                    load(str);
                end
                
            else
                AlldataPts=[];
                tempC=[properties.Centroid];
                AlldataPts(1:length(properties),1)=tempC(1:2:end);
                AlldataPts(1:length(properties),2)=tempC(2:2:end);
                
                dataPts=AlldataPts(:,1:2);
                
                clustCent=dataPts';
                cluster2dataCell=1:length(dataPts);
                %,data2cluster,cluster2dataCell] = MeanShiftCluster(dataPts',bandWidth);
                
                %%%% check
                %%% show the cluster nodes only
                %         LshowBWonIM(mask,I,1);hold on;
                %             for k = 1:length(nuclei)
                %                 if nuclei_label(k)
                %                     plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 1);
                %                 else
                %                     plot(nuclei{k}(:,2), nuclei{k}(:,1), 'b-', 'LineWidth', 1);
                %                 end
                %             end
                %
                %         show(I,1);hold on;
                %         numObj=size(clustCent,2);
                %         for k = 1 : numObj
                %             plot(clustCent(1,k),clustCent(2,k), 'gp');
                %         end
                %
                
            end
            %%% save results
            
            %%  !!!!!!!! below code are for the feature extractions with epi/stroma seperation !!!!!!!!
            %% grouping nuclei based on epi/stroma area, denoting epi/stroma as ES henceforth
            if flag_epistroma
                %             strSaveCur=sprintf('%s%s%s_nuclei.mat',strSavePath,[curPID '/'],curName);
                strSave=sprintf('%s%snuclei_label_in_epistroma_IM%s.mat',strSavePath,[curPID '/'],[]);
                if exist(strSave,'file')~=2
                    %% read ES mask first
                    str_cur=curName;
                    %             str=sprintf('%s%s_mask.png',strEpiStrMaskPath,str_cur(1:end-4));
                    str=sprintf('%s%s.png',strEpiStrMaskPath,curMask);
                    bw=imread(str); %show(bw); show(~mask);show(bw(:,:,3));
                    if size(bw,3)~=1
                        tmp=bw(:,:,3);
                        bw=bw(:,:,3)>range(tmp(:))/2;%show(bw);
                    end
                    %%%% fill in some small holes due to inaccurate annotations.
                    cc= bwconncomp(~bw);
                    stats = regionprops(cc, 'Area');
                    idx = find([stats.Area] > 100);
                    bw_remove = ismember(labelmatrix(cc), idx); %show( bw_remove);
                    mask= ~bw_remove;
                    %%%% project the mask to original dimension
                    mask=imresize(mask,[size(I,1) size(I,2)]);%  LshowBWonIM(mask,I);
                    %show(mask);
                    %% find nuclei in epi/stroma, use 'nuclei_label' to indicate the nuclei
                    %%% in epi-1, or stroma-0
                    all_xy=[properties.Centroid];
                    all_x=all_xy(1:2:end);all_y=all_xy(2:2:end);
                    all_idx=sub2ind([size(I,1) size(I,2)],round(all_y),round(all_x));
                    mask_list=mask(:);
                    nuclei_label=mask_list(all_idx);
                    %% check epi/stroma
%                     LshowBWonIM(mask,I,1);hold on;
%                     for k = 1:length(nuclei)
%                         if nuclei_label(k)
%                             plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 1);
%                         else
%                             plot(nuclei{k}(:,2), nuclei{k}(:,1), 'b-', 'LineWidth', 1);
%                         end
%                     end
%                     hold off;
                    %% Continue
                    save(strSave,'nuclei_label');
                    disp('nuclei label in epi/stroma file saved');
                else
                    disp('nuclei label in epi/stroma file found, loading it');
                    load(strSave);
                end
                %% nuclei cluster finding - meanshift with epi/stroma seperation
                if flag_nuclei_cluster_finding
                    bandWidth=30;
                    str=sprintf('%s%sMSCellCluster_BW%d_IM%s_epistroma.mat',strSavePath,[curPID '/'],bandWidth,curName);
                    if exist(str,'file')~=2
                        try
                            %% use meanshift to form cell clusters, use location info only
                            %%% in epi
                            AlldataPts=[];
                            tempC=[properties(nuclei_label).Centroid];
                            AlldataPts(1:sum(nuclei_label),1)=tempC(1:2:end);
                            AlldataPts(1:sum(nuclei_label),2)=tempC(2:2:end);
                            
                            dataPts_epi=AlldataPts(:,1:2);
                            
                            disp('begin cell cluster finding in epi;');
                            
                            [clustCent_epi,data2cluster_epi,cluster2dataCell_epi] = MeanShiftCluster(dataPts_epi',bandWidth);
                            
                            %%% in stroma
                            AlldataPts=[];
                            tempC=[properties(~nuclei_label).Centroid];
                            AlldataPts(1:sum(~nuclei_label),1)=tempC(1:2:end);
                            AlldataPts(1:sum(~nuclei_label),2)=tempC(2:2:end);
                            
                            dataPts_stroma=AlldataPts(:,1:2);
                            
                            disp('begin cell cluster finding in stroma;');
                            
                            [clustCent_stroma,data2cluster_stroma,cluster2dataCell_stroma] = MeanShiftCluster(dataPts_stroma',bandWidth);
                            %%%% check
                            %%% show the cluster nodes only
                            %             LshowBWonIM(mask,I,1);hold on;
                            % %             for k = 1:length(nuclei)
                            % %                 if nuclei_label(k)
                            % %                     plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 1);
                            % %                 else
                            % %                     plot(nuclei{k}(:,2), nuclei{k}(:,1), 'b-', 'LineWidth', 1);
                            % %                 end
                            % %             end
                            % %
                            %             numObj=size(clustCent_epi,2);
                            %             for k = 1 : numObj
                            %                 plot(clustCent_epi(1,k),clustCent_epi(2,k), 'gp');
                            %             end
                            %
                            %             numObj=size(clustCent_stroma,2);
                            %             for k = 1 : numObj
                            %                 plot(clustCent_stroma(1,k),clustCent_stroma(2,k), 'bp');
                            %             end
                            %             hold off;
                            
                        catch ME
                            idSegLast = regexp(ME.identifier, '(?<=:)\w+$', 'match');
                            %clean up for possible error conditions here. Rethrow if
                            %             unknown error.
                            switch idSegLast{1}
                                case 'nomem'
                                    disp('Out of Memory!');
                                case 'couldNotReadFile'
                                    disp('file not there!');
                                otherwise
                                    %An unexpected error happened
                                    rethrow(ME)
                            end
                            continue
                        end
                        %%% save results
                        
                        save(str,'clustCent_stroma','data2cluster_stroma','cluster2dataCell_stroma',...
                            'clustCent_epi','data2cluster_epi','cluster2dataCell_epi','bandWidth');
                        %
                        %     str=sprintf('%sPro_MSCellCluster_BW%d_%d_IM%d.mat',strPathIM,bandWidth,bandWidth60,i);
                        %     save(str,'clustCent','data2cluster','cluster2dataCell','bandWidth',...
                        %         'clustCent60','data2cluster60','cluster2dataCell60','bandWidth60');
                        disp('nuclei cluster in epi/stroma result saved; ');
                    else
                        disp('nuclei cluster in epi/stroma file found, loading it');
                        load(str);
                    end
                else
                    AlldataPts=[];
                    tempC=[properties(nuclei_label).Centroid];
                    AlldataPts(1:sum(nuclei_label),1)=tempC(1:2:end);
                    AlldataPts(1:sum(nuclei_label),2)=tempC(2:2:end);
                    
                    dataPts_epi=AlldataPts(:,1:2);
                    clustCent_epi=dataPts_epi';
                    cluster2dataCell_epi=1:length(clustCent_epi);
                    
                    %%% in stroma
                    AlldataPts=[];
                    tempC=[properties(~nuclei_label).Centroid];
                    AlldataPts(1:sum(~nuclei_label),1)=tempC(1:2:end);
                    AlldataPts(1:sum(~nuclei_label),2)=tempC(2:2:end);
                    
                    dataPts_stroma=AlldataPts(:,1:2);
                    clustCent_stroma=dataPts_stroma';
                    cluster2dataCell_stroma=1:length(clustCent_stroma);
                    
                    %%%% check
                    %%% show the cluster nodes only
                    %                         LshowBWonIM(mask,I,1);hold on;
                    %             %             for k = 1:length(nuclei)
                    %             %                 if nuclei_label(k)
                    %             %                     plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 1);
                    %             %                 else
                    %             %                     plot(nuclei{k}(:,2), nuclei{k}(:,1), 'b-', 'LineWidth', 1);
                    %             %                 end
                    %             %             end
                    %             %
                    %                         numObj=size(clustCent_epi,2);
                    %                         for k = 1 : numObj
                    %                             plot(clustCent_epi(1,k),clustCent_epi(2,k), 'gp');
                    %                         end
                    %
                    %                         numObj=size(clustCent_stroma,2);
                    %                         for k = 1 : numObj
                    %                             plot(clustCent_stroma(1,k),clustCent_stroma(2,k), 'bp');
                    %                         end
                    %                         hold off;
                    
                    
                end
                %% convert the nuclei bounds into binary mask for some feature extaction
                
                strSave=sprintf('%s%snuclei_binarymask_IM%s.mat',strSavePath,[curPID '/'],[]);
                if exist(strSave,'file')~=2
                    if flag_have_nulei_mask
                        %%% turn the binary imag mask to mat format
                        %bwNuclei=logical(imread([strNucleiMaskPath strrep(curName,'.png','_bwNuc.png')]));%show(bwNuclei)
                        if size(bwNuclei,3)~=1
                            tmp=bwNuclei(:,:,3);
                            bwNuclei=bwNuclei(:,:,3)>range(tmp(:))/2;%show(bw);
                        end
                        % lulu back here
                        %                 error('double check here');
                        str_cur=strrep(curName,'.png','_result.png');
                        %                 str=sprintf('%s%s_mask.png',strEpiStrMaskPath,str_cur(1:end-4));
                        str=sprintf('%s%s.png',strEpiStrMaskPath,str_cur(1:end-4));
                        bw_EP=logical(imread(str)); %show(bw); show(~mask);show(bw(:,:,3));
                        if size(bw_EP,3)~=1
                            tmp=bw_EP(:,:,3);
                            bw_EP=bw_EP(:,:,3)>range(tmp(:))/2;%show(bw_EP);
                        end
                        
                        bwNuclei_epi=logical(bwNuclei)&logical(bw_EP);
                        bwNuclei_stroma=logical(bwNuclei)&~logical(bw_EP);%show(bwNuclei_epi) show(bwNuclei_stroma)
                        
                        save(strSave,'bwNuclei_stroma','bwNuclei_epi','bwNuclei');
                        disp('nuclei binary image file saved');
                    else
                        disp('begin to convert the nuclei bounds to binary mask;');
                        %%% conver the nuclei boundary to binary mask, old code below,
                        %%% slow...
                        %         [m,n,k]=size(I);
                        %         bwNuclei= false(m,n);
                        %         for tt=1:length(nuclei)
                        %             curN=nuclei{tt};
                        %             bw=poly2mask(curN(:,2), curN(:,1),m,n);
                        %             bwNuclei=bwNuclei|bw;
                        %         end
                        
                        %%% conver the nuclei boundary to binary mask
                        [m,n,k]=size(I);
                        bwNuclei=zeros(m,n);
                        for kk = 1:length(nuclei)
                            nuc=nuclei{kk};
                            for kki=1:length(nuc)
                                bwNuclei(nuc(kki,1),nuc(kki,2))=1;
                            end
                        end
                        bwNuclei = imfill(bwNuclei,'holes');
                        %         show(bwNuclei);
                        
                        %%% save nuclei boundary to binary mask in epi/stroma separately
                        nuclei_epi=nuclei(nuclei_label);
                        bwNuclei_epi= false(m,n);
                        for tt=1:length(nuclei_epi)
                            curN=nuclei{tt};
                            for ttt=1:length(curN)
                                bwNuclei_epi(curN(ttt,1),curN(ttt,2))=1;
                            end
                        end
                        bwNuclei_epi = imfill(bwNuclei_epi,'holes');    %         show(bwNuclei_epi);
                        
                        bwNuclei_stroma=bwNuclei-bwNuclei_epi;
                        %                         %% Check
                        %                                     Lshow2BWonIM(bwNuclei,bwNuclei_epi,bwNuclei_stroma,1);
                        %
                        %                                     nuclei_stroma=nuclei(~nuclei_label);
                        %                                     bwNuclei_stroma= false(m,n);
                        %                                     for tt=1:length(nuclei_stroma)
                        %                                         curN=nuclei{tt};
                        %                                         for ttt=1:length(curN)
                        %                                             bwNuclei_stroma(curN(ttt,1),curN(ttt,2))=1;
                        %                                         end
                        %                                     end
                        %                                     bwNuclei_stroma = imfill(bwNuclei_stroma,'holes');show(bwNuclei_stroma);
                        
                        save(strSave,'bwNuclei_stroma','bwNuclei_epi','bwNuclei');
                        disp('nuclei binary image file saved');
                    end
                else
                    disp('nuclei binary image file found, loading it');
                    load(strSave);
                end
                
                %% extract haralick nuclei wise features
                str=sprintf('%s%sHaralickNucleiWiseFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                
                if exist(str,'file')~=2
                    %%
                    disp('begin to exract Haralick Nuclei Wise feature; ');
                    
                    [feats_epi,description_epi] = Lharalick_img_nuclei_wise(rgb2gray(I),bwNuclei_epi,Num_level4Haralick_nuclei_wise);
                    
                    [feats_stroma,description_stroma] = Lharalick_img_nuclei_wise(rgb2gray(I),bwNuclei_epi,Num_level4Haralick_nuclei_wise);
                    
                    save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                    disp('haralick nuclei wise result in epi/stroma saved');
                else
                    disp('haralick nuclei wise file in epi/stroma found');
                    %             load(str);
                end
                
                %% extract Fractal features
                                str=sprintf('%s%sFractalFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                
                                if exist(str,'file')~=2
                                    %%
                                    disp('begin to exract Fractal feature; ');
                
                                    para=[];
                                    [feats_epi,description_epi] = Lextract_Fractal_features(bwNuclei_epi,I,para);
                
                                    [feats_stroma,description_stroma] = Lextract_Fractal_features(bwNuclei_stroma,I,para);
                
                                    save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                                    disp('Fractal features result in epi/stroma saved');
                                else
                                    disp('Fractal features file in epi/stroma found');
                                    %             load(str);
                                end
                
                %% extract CRL features
                                str=sprintf('%s%sCRLFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                
                                para.CGalpha_min=0.38; para.CGalpha_max=0.50;
                                para.alpha_res=0.02;% larger the alpha, sparse the graph
                                para.radius=0.2;
                
                                if exist(str,'file')~=2
                                    %%
                                    disp('begin to exract CRL feature in epi/stroma; ');
                                    %             [I_norm, ~, ~] = normalizeStaining(I);
                
                                    ctemp=[properties(nuclei_label).Centroid];
                                    bounds_epi.centroid_c=ctemp(1:2:end);
                                    bounds_epi.centroid_r=ctemp(2:2:end);
                                    bounds_epi.nuclei=nuclei(nuclei_label);
                
                                    para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                                    para.CCGlocation='epi';
                
                                    [feats_epi,description_epi,~] = Lextract_CRL_features_v2(bounds_epi,I,properties,para);
                
                                    ctemp=[properties(~nuclei_label).Centroid];
                                    bounds_stroma.centroid_c=ctemp(1:2:end);
                                    bounds_stroma.centroid_r=ctemp(2:2:end);
                                    bounds_stroma.nuclei=nuclei(~nuclei_label);
                
                                    para.CCGlocation='stroma';
                
                                    [feats_stroma,description_stroma,~] = Lextract_CRL_features_v2(bounds_stroma,I,properties,para);
                
                                    save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                                    disp('CRL features result in epi/stroma saved');
                                else
                                    disp('CRL features file in epi/stroma found');
                                    %             load(str);
                                end
                
                %% extract basic shape feature in epi/stroma
                str=sprintf('%s%sBasicShapeFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                
                if exist(str,'file')~=2
                    %%
                    disp('begin to exract Basic Shape feature in epi/stroma; ');
                    [feats_epi,description_epi]=Lextract_BasicShape_Features(properties(nuclei_label));
                    [feats_stroma,description_stroma]=Lextract_BasicShape_Features(properties(~nuclei_label));
                    
                    % check properties
                    %         show(I);hold on;
                    %         for k=1:length(properties)
                    %             plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 1);
                    %             temp=properties(k).Centroid;
                    %             text( temp(1),temp(2),num2str(properties(k).Area));
                    %         end
                    %         hold off;
                    
                    save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                    disp('Basic Shape features result in epi/stroma saved');
                else
                    disp('Basic Shape features file in epi/stroma found, loading it');
                    load(str);
                end
                %% extract basic shape feature in CCG in epi/stroma
                str=sprintf('%s%sBasicShapeinCCGFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                if exist(str,'file')~=2
                    %%
                    disp('begin to exract Basic Shape feature in CCG in epi/stroma; ');
                    para.CGalpha_min=0.38; para.CGalpha_max=0.48;
                    para.alpha_res=0.02;% larger the alpha, sparse the graph
                    para.radius=0.2;
                    para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                    
                    ctemp=[properties(nuclei_label).Centroid];
                    bounds_epi.centroid_c=ctemp(1:2:end);
                    bounds_epi.centroid_r=ctemp(2:2:end);
                    bounds_epi.nuclei=nuclei(nuclei_label);
                    para.CCGlocation='epi';
                    [feats_epi,description_epi] = Lextract_BasicShapeinCCG_features_CCGdensityWraper_precomputeCG(bounds_epi,properties,para);
                    
                    ctemp=[properties(~nuclei_label).Centroid];
                    bounds_stroma.centroid_c=ctemp(1:2:end);
                    bounds_stroma.centroid_r=ctemp(2:2:end);
                    bounds_stroma.nuclei=nuclei(~nuclei_label);
                    para.CCGlocation='stroma';
                    [feats_stroma,description_stroma] = Lextract_BasicShapeinCCG_features_CCGdensityWraper_precomputeCG(bounds_stroma,properties,para);
                    
                    save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                    disp('Basic Shape features in CCG in epi/stroma result saved');
                else
                    disp('Basic Shape features in CCG file found, loading it');
                    load(str);
                end
                %% extract all features with epi/stroma seperation
                str=sprintf('%s%sFullFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                
                %%% the parameter to construct CCG matters, so we want to extract a CCG featuers with different
                %%% parameter.
                
                para.CGalpha_min=0.4; para.CGalpha_max=0.5;
                para.CCGalpha_min=0.36; para.CCGalpha_max=0.40; para.alpha_res=0.01;% larger the alpha, sparse the graph
                para.radius=0.2;
                
                if exist(str,'file')~=2
                    disp('begin to exract all feature; ');
                    %% in epi
                    ctemp=[properties(nuclei_label).Centroid];
                    bounds_epi.centroid_c=ctemp(1:2:end);
                    bounds_epi.centroid_r=ctemp(2:2:end);
                    
                    bounds_epi.nuclei=nuclei(nuclei_label);
                    flagNoEpi=0;
                    if ~isempty(bounds_epi.nuclei)&&length(bounds_epi.nuclei)>50
                        %%% for the cell clusters, keep the cluster center contains more than 1 cell as the node in the CCG
                        list_CCGnode=[];%collection of cell cluster (30 pixels as bandwidth)
                        if flag_nuclei_cluster_finding
                            for j=1:length(cluster2dataCell_epi)
                                if ~isempty(cluster2dataCell_epi{j})
                                    list_CCGnode=[list_CCGnode j];
                                end
                            end
                        else
                            list_CCGnode=cluster2dataCell_epi;
                        end
                        
                        bounds_epi.CellClusterC_c=clustCent_epi(1,list_CCGnode);
                        bounds_epi.CellClusterC_r=clustCent_epi(2,list_CCGnode);
                        
                        para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                        para.CCGlocation='epi';
                        
                        [feats_epi,description_epi,CCGinfo_epi,CGinfo_epi] = Lextract_all_features_v2(bounds_epi,para);
                    else
                        %% if there is no epi region in current image, we assgin all feature value =0,
                        %%% however, note that =0 doesn't mean viod, 0 is also a
                        %%% feature value, but we don't have better choice now. %
                        %%% on 2016 Oct 16. We trager different pre-trained
                        %%% classifiers for epi/stroma empty cases. happy!!!
                        %%% Fortunately, 98% of image have stroma region
                        flagNoEpi=1;
                    end
                    %% in stroma
                    ctemp=[properties(~nuclei_label).Centroid];
                    bounds_stroma.centroid_c=ctemp(1:2:end);
                    bounds_stroma.centroid_r=ctemp(2:2:end);
                    
                    bounds_stroma.nuclei=nuclei(~nuclei_label);
                    if ~isempty(bounds_stroma.nuclei)&&length(bounds_stroma.nuclei)>50
                        %%% for the cell clusters, keep the cluster center contains more than 1 cell as the node in the CCG
                        list_CCGnode=[];%collection of cell cluster (30 pixels as bandwidth)
                        if flag_nuclei_cluster_finding
                            for j=1:length(cluster2dataCell_stroma)
                                if ~isempty(cluster2dataCell_stroma{j})
                                    list_CCGnode=[list_CCGnode j];
                                end
                            end
                        else
                            list_CCGnode=cluster2dataCell_stroma;
                        end
                        bounds_stroma.CellClusterC_c=clustCent_stroma(1,list_CCGnode);
                        bounds_stroma.CellClusterC_r=clustCent_stroma(2,list_CCGnode);
                        para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                        para.CCGlocation='stroma';
                        [feats_stroma,description_stroma,CCGinfo_stroma,CGinfo_stroma] = Lextract_all_features_v2(bounds_stroma,para);
                    else
                        %% if there is no stroma region in current image, we assgin all feature value =0,
                        %%% however, note that =0 doesn't mean viod, 0 is also a
                        %%% feature value, but we don't have better choice now. %
                        %%% on 2016 Oct 16. We trager different pre-trained
                        %%% classifiers for epi/stroma empty cases. happy!!!
                        %%% Fortunately, 98% of image have stroma region
                        feats_stroma=cell(1,length(feats_epi));
                        for vii=1:length(feats_epi)
                            feats_stroma{vii}=feats_epi{vii};
                            %                     feats_stroma=feats_epi;
                            description_stroma{vii}=description_epi{vii};
                        end
                        CCGinfo_stroma=0;
                        CGinfo_stroma=0;
                    end
                    
                    if flagNoEpi
                        feats_epi=cell(1,length(feats_stroma));
                        for vii=1:length(feats_stroma)
                            feats_epi{vii}=feats_stroma{vii};
                            %                     feats_stroma=feats_epi;
                            description_epi{vii}=description_stroma{vii};
                        end
                        CCGinfo_epi=0;
                        CGinfo_epi=0;
                    end
                    save(str,'feats_epi','description_epi','CCGinfo_epi','CGinfo_epi','feats_stroma','description_stroma','CCGinfo_stroma','CGinfo_stroma','para','-v7.3');
                    disp('all features result in epi/stroma saved');
                else
                    disp('all features file in epi/stroma found');
                    %             load(str);
                end
                
                %             %% extract TuMop features
                %             bwNuclei_stroma
                
                %% extract Haralick Nuclei wise features in epi/stroma
                str=sprintf('%s%sHaralickNucleiWiseFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                
                if exist(str,'file')~=2
                    %%
                    disp('begin to exract Haralick Nuclei Wise feature in epi/stroma; ');
                    
                    [feats_epi,description_epi] = Lharalick_img_nuclei_wise(rgb2gray(I),bwNuclei_epi,Num_level4Haralick_nuclei_wise);% Check results' shape
                    feats_epi = feats_epi';
                    description_epi = description_epi';
                    
                    
                    [feats_stroma,description_stroma] = Lharalick_img_nuclei_wise(rgb2gray(I),bwNuclei_stroma,Num_level4Haralick_nuclei_wise);% Check results' shape
                    feats_stroma = feats_stroma';
                    description_stroma = description_stroma';
                    
                    save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                    disp('haralick nuclei wise result in epi/stroma saved');
                else
                    disp('haralick nuclei wise file in epi/stroma found');
                    %             load(str);
                end
                %% extract Cytoplasm Haralick Nuclei wise features in epi/stroma
                str=sprintf('%s%sCytoHaralickNucleiWiseFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                bwDilated_epi = im2bw(imdilate(bwNuclei_epi,se));
                bwDilated_stroma = im2bw(imdilate(bwNuclei_stroma,se));
                bwCyto_epi = bwDilated_epi-bwNuclei_epi;
                bwCyto_stroma = bwDilated_stroma-bwNuclei_stroma;
                cytoMskPath = sprintf('%s%sCytoplasmMask_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                save(cytoMskPath,'bwCyto_epi','bwCyto_stroma','-v7.3')
                if exist(str,'file')~=2
                    %%
                    disp('begin to exract Cytoplasm Haralick feature in epi/stroma;');
                    
                    [feats_epi,description_epi] = Lharalick_img_nuclei_wise(rgb2gray(I),bwCyto_epi,Num_level4Haralick_nuclei_wise);% Check results' shape
                    feats_epi = feats_epi';
                    description_epi = char(strcat('Cyto',description_epi'));
                    
                    [feats_stroma,description_stroma] = Lharalick_img_nuclei_wise(rgb2gray(I),bwCyto_stroma,Num_level4Haralick_nuclei_wise);% Check results' shape
                    feats_stroma = feats_stroma';
                    description_stroma = strcat('Cyto',description_stroma')';
                    
                    save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                    disp('Cytoplasm haralick nuclei wise result in epi/stroma saved');
                else
                    disp('Cytoplasm haralick nuclei wise result in epi/stroma saved found');
                    %                 load(str);
                end
                %% extract haralick features
                if flag_haralick
                    str=sprintf('%s%sHaralickFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],curName);
                    
                    if exist(str,'file')~=2
                        %%
                        disp('begin to exract haralick feature; ');
                        [I_norm, ~, ~] = normalizeStaining(I);
                        bounds.nuclei=nuclei(nuclei_label);
                        [feats_epi,description_epi] = Lextract_haralick_features(bounds,I_norm);
                        
                        bounds.nuclei=nuclei(~nuclei_label);
                        [feats_stroma,description_stroma] = Lextract_haralick_features(bounds,I_norm);
                        
                        save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                        disp('haralick features result in epi/stroma saved');
                    else
                        disp('haralick features file in epi/stroma found');
                        %                 load(str);
                    end
                end
                %% extract CCM features
                if flag_CCM
                    %% extract the original cCCM
                    str=sprintf('%s%sCCMFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                    para.CGalpha_min=0.38; para.CGalpha_max=0.5;
                    para.alpha_res=0.02;% larger the alpha, sparse the graph set_alpha=[para.CGalpha_min:para.alpha_res:para.CGalpha_max];
                    para.radius=0.2;
                    if exist(str,'file')~=2
                        %%
                        disp('begin to exract CCM feature in epi/stroma; ');
                        [I_norm, ~, ~] = normalizeStaining(I);
                        
                        if sum(nuclei_label)>10
                            ctemp=[properties(nuclei_label).Centroid];
                            bounds_epi.centroid_c=ctemp(1:2:end);
                            bounds_epi.centroid_r=ctemp(2:2:end);
                            bounds_epi.nuclei=nuclei(nuclei_label);
                            
                            para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                            para.CCGlocation='epi';
                            [feats_epi,description_epi,~] = Lextract_CCM_features_v2(bounds_epi,I_norm,properties,para);%show(I_norm)
                        else
                            set_alpha=[para.CGalpha_min:para.alpha_res:para.CGalpha_max];
                            feats_epi=zeros(1,length(set_alpha)*20*6*13);
                            description_epi='no epi LoCoM features';
                        end
                        
                        if sum (~nuclei_label)>10
                            ctemp=[properties(~nuclei_label).Centroid];
                            bounds_stroma.centroid_c=ctemp(1:2:end);
                            bounds_stroma.centroid_r=ctemp(2:2:end);
                            
                            bounds_stroma.nuclei=nuclei(~nuclei_label);
                            para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                            para.CCGlocation='stroma';
                            [feats_stroma,description_stroma,~] = Lextract_CCM_features_v2(bounds_stroma,I_norm,properties,para);
                        else
                            set_alpha=[para.CGalpha_min:para.alpha_res:para.CGalpha_max];
                            feats_stroma=zeros(1,length(set_alpha)*20*6*13);
                            description_stroma='no stroma LoCoM features';
                        end
                        
                        
                        save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                        disp('CCM features result in epi/stroma saved');
                    else
                        disp('CCM features file in epi/stroma found');
                        %                 load(str);
                    end
                    %% extract the cCCM
                    str=sprintf('%s%scCCMFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],[]);
                    para.CGalpha_min=0.38; para.CGalpha_max=0.5;
                    para.alpha_res=0.02;% larger the alpha, sparse the graph
                    para.radius=0.2;
                    if exist(str,'file')~=2
                        %%
                        disp('begin to exract cCCM feature in epi/stroma; ');
                        [I_norm, ~, ~] = normalizeStaining(I);
                        
                        if sum (nuclei_label)>10
                            ctemp=[properties(nuclei_label).Centroid];
                            bounds_epi.centroid_c=ctemp(1:2:end);
                            bounds_epi.centroid_r=ctemp(2:2:end);
                            bounds_epi.nuclei=nuclei(nuclei_label);
                            
                            para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                            para.CCGlocation='epi';
                            [feats_epi,description_epi,~] = Lextract_CCM_features_v3(bounds_epi,I_norm,properties,para);
                        else
                            set_alpha=para.CGalpha_min:para.alpha_res:para.CGalpha_max;
                            feats_epi=zeros(1,length(set_alpha)*20*6*13);
                            description_epi='no epi LoCoM features';
                        end
                        
                        if sum (~nuclei_label)>10
                            ctemp=[properties(~nuclei_label).Centroid];
                            bounds_stroma.centroid_c=ctemp(1:2:end);
                            bounds_stroma.centroid_r=ctemp(2:2:end);
                            
                            bounds_stroma.nuclei=nuclei(~nuclei_label);
                            para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                            para.CCGlocation='stroma';
                            [feats_stroma,description_stroma,~] = Lextract_CCM_features_v3(bounds_stroma,I_norm,properties,para);
                        else
                            set_alpha=para.CGalpha_min:para.alpha_res:para.CGalpha_max;
                            feats_stroma=zeros(1,length(set_alpha)*20*6*13);
                            description_stroma='no stroma LoCoM features';
                        end
                        
                        save(str,'feats_stroma','description_stroma','feats_epi','description_epi','-v7.3');
                        disp('cCCM features result in epi/stroma saved');
                    else
                        disp('cCCM features file in epi/stroma found');
                        %                 load(str);
                    end
                end
                
                %% extract Hosoya features
                if flag_Hosoya
                    
                    str=sprintf('%s%sHosoyaFeatures_IM%s_epistroma.mat',strSavePath,[curPID '/'],curName);
                    para.CGalpha_min=0.38; para.CGalpha_max=0.48;
                    para.alpha_res=0.02;% larger the alpha, sparse the graph
                    para.radius=0.2;
                    
                    if exist(str,'file')~=2
                        %%
                        disp('begin to exract Hosoya feature in epi/stroma; ');
                        
                        ctemp=[properties(nuclei_label).Centroid];
                        bounds_epi.centroid_c=ctemp(1:2:end);
                        bounds_epi.centroid_r=ctemp(2:2:end);
                        bounds_epi.nuclei=nuclei(nuclei_label);
                        
                        ctemp=[properties(~nuclei_label).Centroid];
                        bounds_stroma.centroid_c=ctemp(1:2:end);
                        bounds_stroma.centroid_r=ctemp(2:2:end);
                        bounds_stroma.nuclei=nuclei(~nuclei_label);
                        
                        para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                        CCGinfo_epi=[];CCGinfo_stroma=[];
                        
                        set_alpha=[para.CGalpha_min:para.alpha_res:para.CGalpha_max];
                        for f=1:length(set_alpha)
                            curpara.alpha=set_alpha(f); curpara.radius=para.radius;
                            para.CCGlocation='epi';
                            CCGinfo_epi=[];
                            if length(bounds_epi.nuclei)>10
                                [hosoya_all_epi{f}, CCGinfo_epi{f}]= Lextract_Hosoya_features_v2(bounds_epi,curpara.alpha,curpara.radius,para);
                            else
                                hosoya_all_epi{f}=0;
                                CCGinfo_epi{f}=0;
                            end
                            
                            para.CCGlocation='stroma';
                            CCGinfo_stroma=[];
                            if length(bounds_epi.nuclei)>10
                                [hosoya_all_stroma{f}, CCGinfo_stroma{f}]= Lextract_Hosoya_features_v2(bounds_stroma,curpara.alpha,curpara.radius,para);
                            else
                                hosoya_all_stroma{f}=0;
                                CCGinfo_stroma{f}=0;
                            end
                        end
                        
                        save(str,'hosoya_all_stroma','hosoya_all_epi','set_alpha','CCGinfo_epi','CCGinfo_stroma','para','-v7.3');
                        clear CCGinfo_epi;clear CCGinfo_stroma;
                        disp('hosoya features result saved');
                    else
                        disp('Hosoya features file found, loading it');
                        load(str);
                    end
                end %strPathIM
                %% below code is for the whole image (no epi/stroma separation)
            else
                
                %% convert the nuclei bonds into binary mask for some feature extaction
                strSave=sprintf('%s%snuclei_binarymask_whole_IM%s.mat',strSavePath,[curPID '/'],curName);
                if exist(strSave,'file')~=2
                    %                 if flag_have_nulei_mask
                    disp('begin to convert the nuclei bounds to binary mask;');
                    %%% conver the nuclei boundary to binary mask, old code below,
                    %%% slow...
                    %         [m,n,k]=size(I);
                    %         bwNuclei= false(m,n);
                    %         for tt=1:length(nuclei)
                    %             curN=nuclei{tt};
                    %             bw=poly2mask(curN(:,2), curN(:,1),m,n);
                    %             bwNuclei=bwNuclei|bw;
                    %         end
                    
                    %%% conver the nuclei boundary to binary mask
                    [m,n,k]=size(I);
                    bwNuclei=zeros(m,n);
                    for kk = 1:length(nuclei)
                        nuc=nuclei{kk};
                        for kki=1:length(nuc)
                            bwNuclei(nuc(kki,1),nuc(kki,2))=1;
                        end
                    end
                    %bwNuclei = imfill(bwNuclei,'holes');
                    %         show(bwNuclei);
                    save(strSave,'bwNuclei');
                    disp('nuclei binary image file saved');
                    %                 else
                    
                    %                 end
                else
                    disp('nuclei binary image file found, loading it');
                    load(strSave);
                    %                 sprintf('%s%snuclei_binarymask_whole_IM%s.mat',strSavePath,[curPID '/'],curName);
                    %                 load ([ curName]);
                    %
                    %                 strSave=sprintf('%s%snuclei_binarymask_IM%s.mat',strSavePath,[curPID '/'],curName);
                    %                 bwNuclei
                    %bwNuclei=logical(imread([strNucleiMaskPath curName]));
                end
                
                %% extract all features
                str=sprintf('%s%sMorphFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                
                %%% the parameter to construct CCG matters, so we want to extract a CCG featuers with different
                %%% parameter.
                
                para.CGalpha_min=0.4; para.CGalpha_max=0.5;
                para.CCGalpha_min=0.36; para.CCGalpha_max=0.40; para.alpha_res=0.01;% larger the alpha, sparse the graph
                para.radius=0.2;
                %% Bounds here
                if exist(str,'file')~=2
                    disp('begin to exract all feature; ');
                    %%
                    ctemp=[properties.Centroid];
                    bounds.centroid_c=ctemp(1:2:end);
                    bounds.centroid_r=ctemp(2:2:end);
                    
                    bounds.nuclei=nuclei;
                    
                    %%% for the cell clusters, keep the cluster center contains more than 1 cell as the node in the CCG
                    list_CCGnode=[];%collection of cell cluster (30 pixels as bandwidth)
                    if flag_nuclei_cluster_finding
                        for j=1:length(cluster2dataCell)
                            if ~isempty(cluster2dataCell{j})
                                list_CCGnode=[list_CCGnode j];
                            end
                        end
                    else
                        list_CCGnode=cluster2dataCell;
                    end
                    
                    bounds.CellClusterC_c=clustCent(1,list_CCGnode);
                    bounds.CellClusterC_r=clustCent(2,list_CCGnode);
                    
                    para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                    para.CCGlocation='whole';
                    %% Morph here with CGT
                    [feats,description,CCGinfo,CGinfo] = Lextract_all_features_v2(bounds,para);
                    %% Morph only
                    %                 feats = extract_morph_feats(bounds);
                    %                 temp = regexp(o.Description.ImageFeatures,'Morph');
                    %                 temp2 = cellfun(@isempty,temp,'UniformOutput',false);
                    %                 description = [o.Description.ImageFeatures([temp2{:}] == 0)];
                    save(str,'feats','description','para','-v7.3');
                    disp('all features result saved');
                else
                    disp('all features file found');
                    %         load(str);
                end
                %% extract haralick features
                if flag_haralick
                    str=sprintf('%s%sHaralickFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                    
                    if exist(str,'file')~=2
                        %%
                        disp('begin to exract haralick feature; ');
                        [I_norm, ~, ~] = normalizeStaining(I);
                        bounds.nuclei=nuclei;
                        [feats,description] = Lextract_haralick_features(bounds,I_norm);% need to reshape the result
                        feats = cellfun(@transpose,feats,'UniformOutput',false);% make it column vector
                        description = cellfun(@transpose,description,'UniformOutput',false);% make it column vector
                        save(str,'feats','description','-v7.3');
                        disp('haralick features result saved');
                    else
                        disp('haralick features file found');
                        %             load(str);
                    end
                end
                %% extract CMM features
                if flag_CCM
                    %% for original CCM
                    str=sprintf('%s%sCCMFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                    para.CGalpha_min=0.38; para.CGalpha_max=0.5;
                    para.alpha_res=0.02;% larger the alpha, sparse the graph
                    para.radius=0.2;
                    
                    if exist(str,'file')~=2
                        %%
                        disp('begin to exract CMM feature; ');
                        [I_norm, ~, ~] = normalizeStaining(I);
                        
                        ctemp=[properties.Centroid];
                        bounds.centroid_c=ctemp(1:2:end);
                        bounds.centroid_r=ctemp(2:2:end);
                        
                        bounds.nuclei=nuclei;
                        
                        para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                        para.CCGlocation='whole';
                        [feats,description,~] = Lextract_CCM_features_v2(bounds,I_norm,properties,para);
                        
                        save(str,'feats','description','-v7.3');
                        disp('CCM features result saved');
                    else
                        disp('CCM features file found');
                        %             load(str);
                    end
                    
                    %% for cCCM
                    str=sprintf('%s%scCCMFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                    para.CGalpha_min=0.38; para.CGalpha_max=0.5;
                    para.alpha_res=0.02;% larger the alpha, sparse the graph
                    para.radius=0.2;
                    
                    if exist(str,'file')~=2
                        %%
                        disp('begin to exract cCMM feature; ');
                        [I_norm, ~, ~] = normalizeStaining(I);
                        
                        ctemp=[properties.Centroid];
                        bounds.centroid_c=ctemp(1:2:end);
                        bounds.centroid_r=ctemp(2:2:end);
                        
                        bounds.nuclei=nuclei;
                        
                        para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                        para.CCGlocation='whole';
                        [feats,description,~] = Lextract_CCM_features_v3(bounds,I_norm,properties,para);
                        
                        save(str,'feats','description','-v7.3');
                        disp('cCCM features result saved');
                    else
                        disp('cCCM features file found');
                        %             load(str);
                    end
                end
                
                %% extract CRL features
                str=sprintf('%s%sCRLFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                para.CGalpha_min=0.38; para.CGalpha_max=0.50;
                para.alpha_res=0.02;% larger the alpha, sparse the graph
                para.radius=0.2;
                
                %             if exist(str,'file')~=2
                %                 %%
                %                 disp('begin to exract CRL feature; ');
                %                 %         [I_norm, ~, ~] = normalizeStaining(I);
                %
                %                 ctemp=[properties.Centroid];
                %                 bounds.centroid_c=ctemp(1:2:end);
                %                 bounds.centroid_r=ctemp(2:2:end);
                %
                %                 bounds.nuclei=nuclei;
                %
                %                 para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                %                 para.CCGlocation='whole';
                %                 [feats,description,~] = Lextract_CRL_features_v2(bounds,I,properties,para);
                %
                %                 save(str,'feats','description','-v7.3');
                %                 disp('CRL features result saved');
                %             else
                %                 disp('CRL features file found');
                %                 %         load(str);
                %             end
                
                %% extract Haralick Nuclei wise features
                str=sprintf('%s%sHaralickNucleiWiseFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                
                if exist(str,'file')~=2
                    %%
                    disp('begin to exract Haralick Nuclei Wise feature; ');
                    
                    [feats,description] = Lharalick_img_nuclei_wise(rgb2gray(I),bwNuclei,Num_level4Haralick_nuclei_wise);% Check results' shape
                    feats = feats';
                    description = description';
                    save(str,'feats','description','-v7.3');
                    disp('haralick nuclei wise result saved');
                else
                    disp('haralick nuclei wise file found');
                    %             load(str);
                end
                %% extract Cytoplasm Haralick Nuclei wise features
                str=sprintf('%s%sCytoHaralickNucleiWiseFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                bwDilated = im2bw(imdilate(bwNuclei,se));
                bwCyto = bwDilated-bwNuclei;
                cytoMskPath = sprintf('%s%sCytoplasmMask_IM%s.mat',strSavePath,[curPID '/'],curName);
                save(cytoMskPath,'bwCyto','-v7.3')
                if exist(str,'file')~=2
                    %%
                    disp('begin to exract Cytoplasm Haralick feature; ');
                    
                    [feats,description] = Lharalick_img_nuclei_wise(rgb2gray(I),bwCyto,Num_level4Haralick_nuclei_wise);% Check results' shape
                    feats = feats';
                    description = char(strcat('Cyto',description'));
                    save(str,'feats','description','-v7.3');
                    disp('Cytoplasm haralick nuclei wise result saved');
                else
                    disp('Cytoplasm haralick nuclei wise file found');
                    %             load(str);
                end
                %% extract Fractal features
                str=sprintf('%s%sFractalFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                %             if exist(str,'file')~=2
                %                 disp('begin to exract Fractal feature; ');
                %                 [feats,description] = Lextract_Fractal_features(bwNuclei,I,para);
                %                 save(str,'feats','description','-v7.3');
                %                 disp('Fractal features result saved');
                %             else
                %                 disp('Fractal features file found');
                %                 %         load(str);
                %             end
                %% extract basic shape feature
                str=sprintf('%s%sBasicShapeFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                if exist(str,'file')~=2
                    %%
                    disp('begin to exract Basic Shape feature; ');
                    [feats,description]=Lextract_BasicShape_Features(properties);
                    % check properties
                    %         show(I);hold on;
                    %         for k=1:length(properties)
                    %             plot(nuclei{k}(:,2), nuclei{k}(:,1), 'g-', 'LineWidth', 1);
                    %             temp=properties(k).Centroid;
                    %             text( temp(1),temp(2),num2str(properties(k).Area));
                    %         end
                    %         hold off;
                    save(str,'feats','description','-v7.3');
                    disp('Basic Shape features result saved');
                else
                    disp('Basic Shape features file found, loading it');
                    load(str);
                end
                
                %% extract basic shape feature in CCG
                str=sprintf('%s%sBasicShapeinCCGFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                %             if exist(str,'file')~=2
                %                 %%
                %                 disp('begin to exract Basic Shape feature in CCG; ');
                %                 ctemp=[properties.Centroid];
                %                 bounds.centroid_c=ctemp(1:2:end);
                %                 bounds.centroid_r=ctemp(2:2:end);
                %                 bounds.nuclei=nuclei;
                %
                %                 para.CGalpha_min=0.38; para.CGalpha_max=0.48;
                %                 para.alpha_res=0.02;% larger the alpha, sparse the graph
                %                 para.radius=0.2;
                %
                %                 para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                %                 para.CCGlocation='whole';
                %                 [feats,description] = Lextract_BasicShapeinCCG_features_CCGdensityWraper_precomputeCG(bounds,properties,para);
                %
                %                 save(str,'feats','description','-v7.3');
                %                 disp('Basic Shape features in CCG result saved');
                %             else
                %                 disp('Basic Shape features in CCG file found, loading it');
                %                 load(str);
                %             end
                
                %% extract Hosoya features
                if flag_Hosoya
                    str=sprintf('%s%sHosoyaFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                    para.CGalpha_min=0.38; para.CGalpha_max=0.48;
                    para.alpha_res=0.02;% larger the alpha, sparse the graph
                    para.radius=0.2;
                    
                    if exist(str,'file')~=2
                        disp('begin to exract Hosoya feature; ');
                        
                        ctemp=[properties.Centroid];
                        bounds.centroid_c=ctemp(1:2:end);
                        bounds.centroid_r=ctemp(2:2:end);
                        bounds.nuclei=nuclei;
                        %             ,[curPID '/']
                        para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                        para.CCGlocation='whole';
                        
                        CCGinfo=[];
                        set_alpha=[para.CGalpha_min:para.alpha_res:para.CGalpha_max];
                        for f=1:length(set_alpha)
                            curpara.alpha=set_alpha(f); curpara.radius=para.radius;
                            [hosoya_all{f}, CCGinfo{f}]= Lextract_Hosoya_features_v2(bounds,curpara.alpha,curpara.radius,para);
                        end
                        
                        save(str,'hosoya_all','set_alpha','CCGinfo','para','-v7.3');
                        clear CCGinfo;
                        disp('hosoya features result saved');
                    else
                        disp('Hosoya features file found, loading it');
                        load(str);
                    end
                end
                %%   extract FeDeG features.
                str=sprintf('%s%sFeDeGFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
                %%% the parameter to construct FeDeG matters, so we want to extract FeDeG featuers with different
                %%% parameter. % note that it also affected by the
                
                para.bandwidth_min=100;% higher the bigger of FeDeG  bandwidth for space only
                para.bandwidth_max=200;% higher the bigger of FeDeG
                para.bandwidth_res=100;
                
                %             if exist(str,'file')~=2
                %                 disp('begin to exract FeDeG features; ');
                %                 % this is not neccesary, since the function only use properties for nuclear features
                %                 [I_norm, ~, ~] = normalizeStaining(I);%show(I_norm)
                %                 ctemp=[properties.Centroid];
                %                 bounds.centroid_c=ctemp(1:2:end);
                %                 bounds.centroid_r=ctemp(2:2:end);
                %
                %                 bounds.nuclei=nuclei;
                %
                %                 %         para.curName=sprintf('%s%s%s',strSavePath,[curPID '/'],curName(1:end-4));
                %                 %         para.CCGlocation='whole';
                %
                %                 %             [feats,description] = Lextract_FeDeG_features_wraper(bounds,I_norm,properties,nuclei,para);
                %                 %%
                %                 %             para.feature_space='Centroid-MeanIntensity';
                %                 para.feature_space='Centroid-Area-MeanIntensity';
                %                 %         para.Feature='Centroid-Area-Eccent-Solid';
                %                 para.debug=0;
                %                 para.nuclei=nuclei;
                %                 para.properties=properties;
                %                 % para.bandwidth=40;
                %                 para.bandWidth_space=200;% higher the bigger of FeDeG
                %                 %             para.bandWidth_features=[20];% bandwidth of in the feature space
                %
                %                 para.bandWidth_features=[60;10];% bandwidth of in the feature space
                %                 % para.bandWidth_features=[20;5];% bandwidth of in the feature space
                %                 para.num_fixed_types=3;
                %                 [feats,description] = Lextract_FeDeG_features_wraper_v2(bounds,I_norm,properties,nuclei,para);
                %
                %                 %             [clustCent,data2cluster,cluster2dataCell,data_other_attribute,clust2types,typeCent]=Lconstruct_FeDeG_v2(I,para);
                %                 %
                %                 %             para.I=I;
                %                 %             para.data_other_attribute=data_other_attribute;
                %                 %             para.debug=0;
                %                 %             para.clust2types=clust2types;
                %                 %             para.typeCent=typeCent;
                %                 %             [feats,description]=L_get_FeDeG_features_v2(clustCent,cluster2dataCell,para);
                %                 %             description=description';
                %
                %                 %% save as mat file
                %                 save(str,'feats','description','-v7.3');
                %                 disp('FeDeG features result saved');
                %             else
                %                 disp('FeDeG features file found, loading it');
                %                 load(str);
                %             end
            end
            %% SpaTIL-V2
%             str=sprintf('%s%sSpaTILFeatures_IM%s.mat',strSavePath,[curPID '/'],curName);
%             if exist(str,'file')~=2
%                 disp('begin to exract SpaTIL feature; ');
%                 [nucleiCentroids,nucFeatures] = getNucLocalFeatures(I,bwNuclei);
%                 model=load('F:\CWRU-Research\Feature Extract Code\SpaTIL_v2\example_data\lymp_svm_matlab_40x.mat');
%                 [label,~,~] = predict(model.model,nucFeatures(:,1:7));% If label == 1, cell is a TIL
%                 isLymphocyte_default = label == 1;% Default thershold
%                 coords={nucleiCentroids(isLymphocyte_default,:),nucleiCentroids(~isLymphocyte_default,:),};
%                 [feats,description]=getSpaTILFeatures_v2(coords);
%                 description = strcat('SpaTIL_',description);
%                 save(str,'feats','description','-v7.3');
%                 disp('SpaTIL features result saved');
%             else
%                 disp('SpaTIL features file found, loading it');
%                 load(str);
%             end
%             
            %% create a file to contain all features and names for an image
            cur_tempmat=sprintf('%s%s*%s*.mat',strSavePath,[curPID '/'],'Features');%curName(1:end-length(image_format))
            %                 cur_tempmat='F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\test\*20150528_191353_x38913_y10241*.mat';
            % curName='20150528_191353_x38913_y10241.png';
            dir_feat_mat=dir(cur_tempmat);
            for i_mat= 1:length(dir_feat_mat)
                cur_feat_mat=dir_feat_mat(i_mat).name;
                if ~contains(cur_feat_mat,'FullFeat')
                    variableInfo = who('-file', [strSavePath [curPID '/'] cur_feat_mat]);
                    %                 variableInfo = who('-file', ['F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\test\' cur_feat_mat]);
                    if ismember('feats', variableInfo)
                        load([strSavePath [curPID '/'] cur_feat_mat]);
                        %                 load(['F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\test\' cur_feat_mat]);
                        if  iscell(feats)
                            feats = [feats{:}];
                            
                        end
                        if  length(description)==1
                            description = description{1};
                        end
                        if size(feats,1)<size(feats,2)% assume the feature is a column vector
                            
                            feats=feats';
                        end
                        
                        allFeats=cat(1,allFeats,feats);
                        if exist(strAllFeatsMat_name,'file')~=2
                            
                            if size(description,1)<size(description,2)% assume the feature is a column vector
                                description=description';
                            end
                            alldescription=cat(1,alldescription,description);
                        end
                    end
                    
                    if ismember('feats_epi', variableInfo)%|ismember('feats_stroma', variableInfo)
                        load([strSavePath [curPID '/'] cur_feat_mat]);
                        %                 load(['F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\test\' cur_feat_mat]);
                        if size(feats_epi,1)<size(feats_epi,2)% assume the feature is a column vector
                            feats_epi=feats_epi';
                            feats_stroma=feats_stroma';
                        end
                        allFeats=cat(1,allFeats,feats_epi);
                        allFeats=cat(1,allFeats,feats_stroma);
                        
                        if exist(strAllFeatsMat_name,'file')~=2
                            idx=1;feature_list_epi=[];feature_list_stroma=[];
                            for i_des=1:length(description_stroma)
                                %                     curC=description_epi{i_des};
                                %                     for j=1:length(curC)
                                if iscell(description_epi)
                                    feature_list_epi{idx}=['EP:' description_epi{i_des}];
                                else
                                    feature_list_epi{idx}=['EP:' description_stroma{i_des}];
                                end
                                
                                if iscell(description_stroma)
                                    feature_list_stroma{idx}=['ST:' description_stroma{i_des}];
                                else
                                    feature_list_stroma{idx}=['ST:' description_epi{i_des}];
                                end
                                
                                idx=idx+1;
                                %                     end
                            end
                            if size(feature_list_epi,1)<size(feature_list_epi,2)% assume the feature is a column vector
                                feature_list_epi=feature_list_epi';
                                feature_list_stroma=feature_list_stroma';
                            end
                            alldescription=cat(1,alldescription,feature_list_epi);
                            alldescription=cat(1,alldescription,feature_list_stroma);
                        end
                    end
                else
                    variableInfo = who('-file', [strSavePath [curPID '/'] cur_feat_mat]);
                    %                 variableInfo = who('-file', ['F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\test\' cur_feat_mat]);
                    if ismember('feats', variableInfo)
                        load([strSavePath [curPID '/'] cur_feat_mat]);
                        %                 load(['F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\test\' cur_feat_mat]);
                        feats=cell2mat(feats);
                        if size(feats,1)<size(feats,2)% assume the feature is a column vector
                            feats=feats';
                        end
                        allFeats=cat(1,allFeats,feats);
                        
                        if exist(strAllFeatsMat_name,'file')~=2
                            idx=1;feature_list=[];
                            for i_ttt=1:length(description)
                                curC=description{i_ttt};
                                for j=1:length(curC)
                                    feature_list{idx}=curC{j}; idx=idx+1;
                                end
                            end
                            if size(feature_list,1)<size(feature_list,2)% assume the feature is a column vector
                                feature_list=feature_list';
                            end
                            alldescription=cat(1,alldescription,feature_list);
                        end
                    end
                    
                    if ismember('feats_epi', variableInfo)%|ismember('feats_stroma', variableInfo)
                        load([strSavePath [curPID '/'] cur_feat_mat]);
                        %                 load(['F:\Nutstore\Nutstore\PathImAnalysis_Program\Program\Features\test\' cur_feat_mat]);
                        feats_epi=cell2mat(feats_epi);
                        feats_stroma=cell2mat(feats_stroma);
                        if size(feats_epi,1)<size(feats_epi,2)% assume the feature is a column vector
                            feats_epi=feats_epi';
                            feats_stroma=feats_stroma';
                        end
                        allFeats=cat(1,allFeats,feats_epi);
                        allFeats=cat(1,allFeats,feats_stroma);
                        
%                         if exist(strAllFeatsMat_name,'file')~=2
%                             idx=1;feature_list_epi=[];feature_list_stroma=[];
%                             for i_des=1:18%length(description_stroma)
%                                 curC=description_stroma{i_des};
%                                 for j=1:length(curC)
%                                     feature_list_epi{idx}=['EP:' curC{j}];
%                                     feature_list_stroma{idx}=['ST:' curC{j}];
%                                     idx=idx+1;
%                                     %                     end
%                                 end
%                             end
%                             if size(feature_list_epi,1)<size(feature_list_epi,2)% assume the feature is a column vector
%                                 feature_list_epi=feature_list_epi';
%                                 feature_list_stroma=feature_list_stroma';
%                             end
%                             alldescription=cat(1,alldescription,feature_list_epi);
%                             alldescription=cat(1,alldescription,feature_list_stroma);
%                         end
                    end
                end
                delete([strSavePath [curPID '/'] cur_feat_mat]);
            end
            %%% delete all temper files
            %         eval(sprintf('delete %s', cur_tempmat));
            %         disp('deleting all temporate files');
            
            save(strAllFeatsMat,'allFeats','-v7.3');
            disp('all feature saved');
%             if exist(strAllFeatsMat_name,'file')~=2
%                 save(strAllFeatsMat_name,'alldescription','-v7.3');
%             end
            %         clear allFeats
            %         clear alldescription
        end
        fprintf('The mask idx is %d, job done. Image name is %s,\n',i,curName);
    catch
%         fid = fopen(fullfile(pwd, strcat('featExtraction-Error-',date,'.txt'))...
%             ,'a');
%         fprintf(fid,'The mask idx is %d, job Error. Image name is %s,\n',i,curName);
%         fclose(fid);
     end
end

