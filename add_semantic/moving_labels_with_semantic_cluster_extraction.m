% Parameters of cluster extraction
minDistance = 0.5;
minPoints = 10;
% Percentage of minimum moving labeled points of an object to consider as
% moving object
dynthreshpercent = 0.4;
% Minimum points to consider point cluster as object
threshpoint = 200;


% Path for velodyne files
velo_dir = 'G:\Virtualbox\VBshared\kitti_raw\2011_09_30_drive_0028_sync\velodyne_points\data';
% Path for moving labels
pred_movlab = 'E:/Kitti_odometry/2011_09_30_drive_0028_sync/pred-merged_2dpass-sparse-vote1_mos-frame2\sequences\08\predictions\';
% Path for labels of semantic segmentation
sem_labels = 'C:\Users\Zoli\Desktop\New_folder\salsanext-semantic\sequences\08\predictions\';

% Output path for new labels
labels_out = 'C:\New_labels\sequences\08\predictions\';

% Iterating through toy dataset
for ii=0:4070
    
    % Read velodyne files and store into Matlab point Cloud format
    velo_name = [velo_dir,'\',sprintf('%06d',ii),'.bin'];
    fd = fopen(velo_name,'rb');
    velo1 = fread(fd,[4 inf],'single')';
    fclose(fd);
    xyz = velo1(:,1:3);
    ptCloud = pointCloud(velo1(:,1:3));
    ptCloud.Intensity = velo1(:,4);

    % Read predicted moving object labels and find them
    [pc_lb_mov, pc_id_mov] = readLabel([pred_movlab,sprintf('%06d',ii),'.label']);
    pred_lab = find(pc_lb_mov>250);

    % Read predicted semantic labels and find them
    [pc_lb_sm, pc_id_sm] = readLabel([sem_labels,sprintf('%06d',ii),'.label']);
    carIdx = find(pc_lb_sm == 10);

    % Select the points of required class and cluster them based on distance.
    ptCldMod = select(ptCloud,carIdx);
    [labels,numClusters] = pcsegdist(ptCldMod,minDistance,"NumClusterPoints",minPoints);
    
    % Select each cluster and fit a cuboid to each cluster.
    pred_lab_up = [];
    is_dyn = zeros(numClusters,1);
    for num = 1:numClusters
        labelIdx = (labels == num);
        
        % Ignore cluster that has points less than threshpoint points.
        if sum(labelIdx,'all') < threshpoint
            continue;
        end
        
        % Count how many percent of the given instance has moving label 
        is_dyn = length(intersect(pred_lab,carIdx(labelIdx)))/length(carIdx(labelIdx));
        
        % Label all points of the instance as moving, if the moving point
        % percent above threshold
        if is_dyn>dynthreshpercent
            pred_lab_up = [pred_lab_up carIdx(labelIdx)];
        end

    end

    % Write the new labels into files
    pred_lab_up = unique([pred_lab pred_lab_up]);
    new_labels = 9*ones(1,size(pc_lb_mov,2));
    dontcare = find(pc_lb_mov==0);
    new_labels(dontcare) = 0;
    new_labels(pred_lab_up) = 251;
    fileID = fopen([labels_out,sprintf('%06d',ii),'.label'],'w');
    fwrite(fileID,new_labels,'uint32');
    fclose(fileID);

end