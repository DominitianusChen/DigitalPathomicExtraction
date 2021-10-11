clc
clear
close all
%% Extract Cytoplasm Haralick Features
%mskMainPath = "F:\CWRU-Research\StageIII-Colon\Patches\";
dataset = ["UH";"TCGA"];
for ds = 1:2
    switch ds
        case 1
            savePath = "E:\StageIII-Colon\FeatsAfterStainN\UH\";
            fileMainPath = "H:\Colon-Stg3\normalised\UH\Patch\";
            epiMainPath = "H:\Colon-Stg3\UH\Epi\";
            nucMainPath = 'E:\Hovernet\Stg3\UH\';
        case 2
            savePath = "E:\StageIII-Colon\FeatsAfterStainN\TCGA\";
            fileMainPath = "E:\StageIII-Colon\normalised\CorrectedPatch10072021\";
            epiMainPath = "E:\StageIII-Colon\CorrectedEpi\";
            nucMainPath = 'E:\StageIII-Colon\CorrectedHoverNuc\TCGA\';
    end
    dirFile = dir(fileMainPath);
    dirFile = dirFile(3:end);
    
    %% 20 x
    % imgDir = dir('E:\tumorSeg\TCGA-COAD-Stg3\**\*.svs');
    % imgDir = imgDir(2:end);
    % name20X = string([]);
    % for i =1:length(imgDir)
    %
    %     info = imfinfo(fullfile(imgDir(i).folder,imgDir(i).name));
    %     scan = string(extractBetween(info(1).ImageDescription,'Mag = ','|'));
    %     if strcmp(scan,"20")
    %         tt = extractBefore(imgDir(i).name,'.svs');
    %         name20X = [name20X;tt];
    %     end
    %
    % end
    % [~,~,idxb]=intersect(name20X,string({dirFile.name}'));
    % dirFile = dirFile(idxb);
    %% exrtract
    parfor i = 1:length(dirFile)
        fprintf(sprintf('Working on %s \n',dirFile(i).name))
        strPathIM=char(strcat(fileMainPath,'\',dirFile(i).name,'\'));%important Lextractfeature_v17 only take char array
        strSavePath=char(strcat(savePath,dirFile(i).name,'\'));
        LcreateFolder(strSavePath);
        strEpiStrMaskPath=char(strcat(epiMainPath,'\',dirFile(i).name,'\'));
        strNucleiMaskPath=char(strcat(nucMainPath,'\',dirFile(i).name,'\'));
        if ds == 1
            strSavePath= strrep(strSavePath,'_',' ');
            strEpiStrMaskPath= strrep(strEpiStrMaskPath,'_',' ');
            %strNucleiMaskPath= strrep(strNucleiMaskPath,'_',' ');
        end
        flag_nuclei_cluster_finding=0;
        image_format='.png';
        para_nulei_scale_low=8;
        para_nulei_scale_high=18;
        flag_have_nulei_mask=1;
        flag_epistroma=1;
        strWSIPath='';
        strWSI_format=image_format;
        idxBegin=1;
        %idxEnd=idxBegin;
        idxEnd=length(dirFile);
        flag_Hosoya = 0;% proved useless
        flag_haralick = 0;% Switch to faster version
        flag_CCM = 1;
        flag_WSI = 0;
        flagRandomPatch = [0,floor(0.15*length(dir(strPathIM)))];% Random Pick 15% patchs Modify by Chuheng
        flagRandomPatch = [0,1];
        Lextractfeature_v17(idxBegin,idxEnd,strWSIPath,flag_epistroma,strPathIM,strSavePath,...
            strEpiStrMaskPath,strNucleiMaskPath,flag_have_nulei_mask,flag_nuclei_cluster_finding,...
            image_format,flag_Hosoya,flag_haralick,flag_CCM,flag_WSI,para_nulei_scale_low,para_nulei_scale_high,flagRandomPatch)
    end
end
