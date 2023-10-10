clear all;
clc

% Get the letter/digit
target = input('Goal coordinate on grid: \n', 's');
split_target = split(target,[" "]);
target1 = split_target(1); % num1 string type
target2 = split_target(2); % num2 string type
% Calculate the numbers
goal_i = str2double(target1);
goal_j = str2double(target2);

%% Get webcam
camList = webcamlist
% Connect to the webcam
cam = webcam(1);
% preview(cam);

set(gcf,'currentchar',' ')

while get(gcf,'currentchar')==' '
    % Get image from webcam
    img = snapshot(cam);
    imshow(img);
end
rgb_img = img;
%% Disconnect webcam
clear cam

%% Read the input image from file
% rgb_img = imread('lab3.jpg');

%% Normalise RGB image
norm_img = uint8(double(rgb_img) .* (128 ./ mean(mean(rgb_img, 1), 2)));
figure(1)
imshow(norm_img)
hold on;

% Convert to HSV
hsv_img = rgb2hsv(norm_img);
% figure(2)
% imshow(hsv_img)
% hold on;

%% Correct map for 'lab8.jpg'
% correct_map = [
%     1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
%     1, 0, 0, 0, 1, 0, 0, 0, 3, 1;
%     1, 0, 0, 0, 0, 0, 2, 2, 0, 1;
%     1, 0, 0, 1, 0, 1, 2, 2, 0, 1;
%     1, 0, 0, 2, 0, 0, 0, 0, 0, 1;
%     1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
%     1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
% ];
%% Correct map for 'lab9.jpg'
% correct_map = [
%     1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
%     1, 0, 0, 2, 1, 0, 0, 0, 0, 1;
%     1, 1, 0, 0, 2, 0, 0, 2, 0, 1;
%     1, 0, 0, 0, 0, 0, 0, 2, 3, 1;
%     1, 0, 0, 2, 0, 0, 0, 0, 0, 1;
%     1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
%     1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
% ];
%% Correct map for 'lab2.jpg'
correct_map = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
    1, 0, 0, 0, 0, 2, 0, 0, 3, 1;
    1, 0, 0, 0, 0, 0, 0, 0, 0, 1;
    1, 0, 0, 2, 0, 0, 0, 2, 0, 1;
    1, 0, 1, 1, 0, 1, 2, 0, 0, 1;
    1, 0, 0, 0, 0, 0, 1, 0, 0, 1;
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1;
];

%% Color parameters
% Reference (purple)
ref_h_l = 0.7;
ref_h_h = 0.8;
ref_s_l = 0.6;
ref_s_h = 0.8;
ref_v_l = 0.35;
ref_v_h = 0.7;

% Board corner (pink)
cor_h_l = 0.83;
cor_h_h = 0.95;
cor_s_l = 0.65;
cor_s_h = 0.85;
cor_v_l = 0.7;
cor_v_h = 1.1;

% Red piece
r_h_l = 0.9;
r_h_h = 1;
r_s_l = 0.6;
r_s_h = 0.95;
r_v_l = 0.5;
r_v_h = 0.7;

% Blue piece
b_h_l = 0.6;
b_h_h = 0.7;
b_s_l = 0.8;
b_s_h = 1.1;
b_v_l = 0.65;
b_v_h = 0.9;

% Green piece
g_h_l = 0.35;
g_h_h = 0.45;
g_s_l = 0.75;
g_s_h = 1.1;
g_v_l = 0.5;
g_v_h = 0.65;

%% Find the reference points
% Mask the reference points
mask_ref = (hsv_img(:,:,1)>ref_h_l)&(hsv_img(:,:,1)<ref_h_h)...
    &(hsv_img(:,:,2)>ref_s_l)&(hsv_img(:,:,2)<ref_s_h)...
    &(hsv_img(:,:,3)>ref_v_l)&(hsv_img(:,:,3)<ref_v_h);

% Remove small artifacts containing fewer than 10 pixels
mask_ref = bwareaopen(mask_ref, 10);
% Fill black gaps using structuring element
se = strel('disk', 1);
mask_ref = imdilate(mask_ref, se);
% imshow(mask_ref)

% Detect circles in radii in range[,]
[centers_ref, radii_ref] = imfindcircles(mask_ref, [5,15]);

% Eliminate the centers that are too close, should be equal to 4 reference points
references = [];
if size(centers_ref, 1) > 4
    centers_ref = sortrows(centers_ref);
    for i = 1:size(centers_ref, 1)-1
        if norm(centers_ref(i, :)-centers_ref(i+1, :)) > 80
            references = [references; centers_ref(i, :)];
        end
    end
    references = [references; centers_ref(size(centers_ref, 1), :)];
else
    references = sortrows(centers_ref);
end

% Rearrange the reference points for further transformation
low_refs = [];
high_refs = [];
for i = 1:4
    if references(i, 2) > 200
        low_refs = [low_refs; references(i, :)];
    else
        high_refs = [high_refs; references(i, :)];
    end
end
centers_ref = [sortrows(low_refs); sortrows(high_refs, -1)];

% Show the reference points
plot(centers_ref(:,1), centers_ref(:,2),'b*');

%% Perspective transform
% Find transformation
% Matlab Window
window = [0 600; 0 0;650 0; 650 600];
% Real Coordinates
world = [-900 75; -900 -525; -250 -525; -250 75];
%%%% Calculate the transformation matrix %%%%
img2world_tform = fitgeotrans(centers_ref, world, 'projective');

%% Objective 1: Identify the Four Corners and of the Game Board
% Mask the board corner
% Orange
% mask_o = (norm_img(:,:,1)>cor_r_l)&(norm_img(:,:,1)<cor_r_h)...
%     &(norm_img(:,:,2)>cor_g_l)&(norm_img(:,:,2)<cor_g_h)...
%     &(norm_img(:,:,3)>cor_b_l)&(norm_img(:,:,3)<cor_b_h);
% Pink
mask_o = (hsv_img(:,:,1)>cor_h_l)&(hsv_img(:,:,1)<cor_h_h)...
    &(hsv_img(:,:,2)>cor_s_l)&(hsv_img(:,:,2)<cor_s_h)...
    &(hsv_img(:,:,3)>cor_v_l)&(hsv_img(:,:,3)<cor_v_h);

% Remove small artifacts containing fewer than 10 pixels
mask_o = bwareaopen(mask_o, 10);
% Fill black gaps using structuring element
se = strel('disk', 5);
mask_o = imdilate(mask_o, se);
% imshow(mask_o)

% Detect circles in radii in range[2,10]
[centers_o, radii_o] = imfindcircles(mask_o, [1,10]);

% Eliminate the centers that are too close, should be equal to 4 corners
corners_img = [];
if size(centers_o, 1) > 4
    centers_o = sortrows(centers_o);
    for i = 1:size(centers_o, 1)-1
        if norm(centers_o(i, :)-centers_o(i+1, :)) > 80
            corners_img = [corners_img; centers_o(i, :)];
        end
    end
    corners_img = [corners_img; centers_o(size(centers_o, 1), :)];
else
    corners_img = sortrows(centers_o);
end

% Rearrange the corner points for further transformation
low_cors = [];
high_cors = [];
for i = 1:4
    if corners_img(i, 2) > 200
        low_cors = [low_cors; corners_img(i, :)];
    else
        high_cors = [high_cors; corners_img(i, :)];
    end
end
corners_img = [sortrows(low_cors); sortrows(high_cors, -1)];
plot(corners_img(:,1), corners_img(:,2),'r*');

% Transfer corners to world coordinate
% world_coor = transformPointsForward(tform, img_coor)
[x,y] = transformPointsForward(img2world_tform, corners_img(:,1),corners_img(:,2));
corners_world = [x, y];


% Plot the corners and the world coordinate
for i = 1:size(corners_world, 1)
    text(corners_img(i,1)-10, corners_img(i,2)+10, '(' + string(corners_world(i,1)) + ' ,' + string(corners_world(i,2)) + ')','Color','r')
end

%% Objective 2: Identify Location of Obstacle and Player Pieces on the Gameboard Using Colour Space Image Segmentation Techniques
% Virtual Grid Coordinates
grid_coor = [0 0; 10 0; 10 16; 0 16];
%%%% Calculate the transformation matrix %%%%
grid2world_tform = fitgeotrans(grid_coor, corners_world, 'projective');
grid2img_tform = fitgeotrans(grid_coor, corners_img, 'projective');

% Get all points and colors in the grid
map = [];
% Initialise start position in grid
start_i = 0;
start_j = 0;

for i = 1:2:9
    col = [];
    for j = 1:2:15
        if (i == goal_i)&(j == goal_j)
            [goal_x, goal_y] = transformPointsForward(grid2world_tform,i,j);
        end

        % Get the color
        [x_img,y_img] = transformPointsForward(grid2img_tform,i,j);
        index = round([x_img,y_img]);
        H = 0;
        S = 0;
        V = 0;
        square_size = 6; % must be even number
        for m = 1:square_size
            for n = 1:square_size
                color = hsv_img(index(2)-(square_size/2)+m, index(1)-(square_size/2)+n, :);
                H = H + color(1);
                S = S + color(2);
                V = V + color(3);
            end
        end
        H = H/(square_size^2);
        S = S/(square_size^2);
        V = V/(square_size^2);
        % For getting the color at a specific point
%         if i==1 && j==1
%            x_img
%            y_img
%            H
%            S
%            V
%         end
        
        % Red
        if (H>r_h_l)&&(H<r_h_h)&&(S>r_s_l)&&(S<r_s_h)&&(V>r_v_l)&&(V<r_v_h)
            plot(x_img,y_img,'g*');
            text(x_img+5,y_img,'R');
            c = 1;
        % Blue
        elseif (H>b_h_l)&&(H<b_h_h)&&(S>b_s_l)&&(S<b_s_h)&&(V>b_v_l)&&(V<b_v_h)
            plot(x_img,y_img,'g*');
            text(x_img+5,y_img,'B');
            c = 2;
        % Green (Player)
        elseif (H>g_h_l)&&(H<g_h_h)&&(S>g_s_l)&&(S<g_s_h)&&(V>g_v_l)&&(V<g_v_h)
            plot(x_img,y_img,'g*');
            text(x_img+5,y_img,'G');
            c = 3;
            start_i = i;
            start_j = j;
            [start_x, start_y] = transformPointsForward(grid2world_tform,i,j);
        % Empty
        else
            c = 0;
        end
        col = [c; col];
    end
    map = [map, col];
end

humanView_map = rot90(map, -1);
%% Objective 3: Calculate Homography Matrix and Perform Projective Transform
board_width = round(norm(corners_world(1,:)-corners_world(2,:)));
board_height = round(norm(corners_world(2,:)-corners_world(3,:)));
window = [0 board_width; 0 0;board_height 0; board_height board_width];
%%%% Calculate the transformation matrix %%%%
img2window_tform = fitgeotrans(corners_img, window, 'projective');
% Show transformed frame
figure(3)
board_img = imwarp(rgb_img, img2window_tform, 'OutputView', imref2d([board_width,board_height,3]));
imshow(board_img)
set(gca,'YDir','normal')

%% Part B: Move the Obstacle Pieces to Their Correct Locations and Implement the Bug2
% Determine which blue pieces are in wrong place and assign to the correct place
current_map = padarray(humanView_map, [1, 1], 1)
correct_map
empty_positions = [];
wrongBlue_position = [];
for i = 1:size(current_map, 1)
    for j = 1:size(current_map, 2)
        if current_map(i, j) ~= correct_map(i, j)
            % Get empty position
            if current_map(i, j) == 0
                empty_positions = [empty_positions; [i j]];
            % Get blue pieces in wrong position
            else
                wrongBlue_position = [wrongBlue_position; [i j]];
            end
        end
    end
end

% If there are blue pieces in wrong positions
if ~isequal(empty_positions, [])
    % Transform map(after padding) array index to grid coordinate
    empty_positions = empty_positions*2-3;
    wrongBlue_position = wrongBlue_position*2-3;
    % Transform grid coordinate to world coordinate
    [x_world,y_world] = transformPointsForward(grid2world_tform, empty_positions(:, 1), empty_positions(:, 2));
    empty_world = [x_world,y_world];
    [x_world,y_world] = transformPointsForward(grid2world_tform, wrongBlue_position(:, 1), wrongBlue_position(:, 2));
    wrongBlue_world = [x_world,y_world];
end

%% Generate the Bug2 path to the goal
% Transform grid coordinate to map(after padding) array index
start_i = (start_i+3)/2;
start_j = (start_j+3)/2;
goal_i = (goal_i+3)/2;
goal_j = (goal_j+3)/2;
start = [start_i start_j]
goal = [goal_i goal_j]

% Use correct_map for path generation
correct_map = correct_map(2:size(correct_map, 1)-1, 2:size(correct_map, 2)-1);
% Preprocess map array, set start point to 0 and all obstacles to 4
correct_map(start_i-1, start_j-1) = 0;
correct_map(correct_map ~= 0) = 4;
% Pad the map with 1 (walls)
correct_map = padarray(correct_map, [1, 1], 1);
figure(4)
imshow(correct_map, 'InitialMagnification', 4000);
hold on

% % Preprocess map array, set start point to 0 and all obstacles to 4
% humanView_map(start_i-1, start_j-1) = 0;
% humanView_map(humanView_map ~= 0) = 4;
% % Pad the map with 1 (walls)
% humanView_map = padarray(humanView_map, [1, 1], 1);
% figure(4)
% imshow(humanView_map, 'InitialMagnification', 4000);
% hold on

% Get Bug2 path by map array
bug2Path_map = getBug2Path(correct_map, start, goal)
plot(bug2Path_map(:, 2), bug2Path_map(:, 1),'LineWidth', 4, 'Color', 'g');
% Transform map(after padding) array index to grid coordinate
bug2Path_grid = bug2Path_map*2-3;

% Transform grid coordinate to image coordinate
[x_img,y_img] = transformPointsForward(grid2img_tform, bug2Path_grid(:, 1), bug2Path_grid(:, 2));
figure(1), plot(x_img, y_img,'LineWidth', 4, 'Color', 'g');

% Transform grid coordinate to world coordinate
[x_world,y_world] = transformPointsForward(grid2world_tform, bug2Path_grid(:, 1), bug2Path_grid(:, 2));
bug2Path_world = [x_world,y_world];

%% Start the robot
startup_rvc;
% TCP Host and Port settings
% host = '127.0.0.1'; % THIS IP ADDRESS MUST BE USED FOR THE VIRTUAL BOX VM
host = '192.168.0.100'; % THIS IP ADDRESS MUST BE USED FOR THE REAL ROBOT
rtdeport = 30003;
vacuumport = 63352;
% Calling the constructor of rtde to setup tcp connction
rtde = rtde(host,rtdeport);
vacuum = vacuum(host,vacuumport);

home_position = [-588.53, -133.30, 371.91, 2.2214, -2.2214, 0.00];
zaxis_zero = 8;% Real robot = 8
zaxis_high = zaxis_zero + 20;
% setting move parameters
v = 0.4; % Real robot = 0.1
a = 0.5; % Real robot = 0.8
blend = 0.005; % 0.005
poses = [];
pose = rtde.movej(home_position);

% If there are blue pieces in wrong positions
if ~isequal(empty_positions, [])
    % Move the Obstacle Pieces to Their Correct Locations
    for i = 1:size(wrongBlue_world, 1)
        % Move to wrong blue
        point = [[wrongBlue_world(i, :), zaxis_high], (home_position(4:6))];
        pose = rtde.movej(point);
        poses = cat(1,poses,pose);
        point = [[wrongBlue_world(i, :), zaxis_zero], (home_position(4:6))];
        pose = rtde.movej(point);
        poses = cat(1,poses,pose);
        vacuum.grip()
        pause(3)
        point = [[wrongBlue_world(i, :), zaxis_high], (home_position(4:6))];
        pose = rtde.movej(point);
        poses = cat(1,poses,pose);
        % Move to correct position
        point = [[empty_world(i, :), zaxis_high], (home_position(4:6))];
        pose = rtde.movej(point);
        poses = cat(1,poses,pose);
        point = [[empty_world(i, :), zaxis_zero], (home_position(4:6))];
        pose = rtde.movej(point);
        poses = cat(1,poses,pose);
        vacuum.release()
        point = [[empty_world(i, :), zaxis_high], (home_position(4:6))];
        pose = rtde.movej(point);
        poses = cat(1,poses,pose);
    end
end

% Move to starting point (green piece)
player_pt = [[start_x, start_y, zaxis_high], (home_position(4:6))];
pose = rtde.movej(player_pt);
poses = cat(1,poses,pose);
player_pt = [[start_x, start_y, zaxis_zero], (home_position(4:6))];
pose = rtde.movej(player_pt);
poses = cat(1,poses,pose);
% Turn on vacuum gripper
vacuum.grip()
disp("Grip")
pause(3)
% Move up
player_pt = [[start_x, start_y, zaxis_high], (home_position(4:6))];
pose = rtde.movej(player_pt);
poses = cat(1,poses,pose);

% Move along the path
% Greater than one step
if size(bug2Path_world, 1) > 2
    bug2Path_robot = [];
    for i = 2:size(bug2Path_world, 1)
        point = [[[bug2Path_world(i, :), zaxis_high], (home_position(4:6))],a,v,0,blend];
        if isempty(bug2Path_robot)
            bug2Path_robot = point;
        else
            bug2Path_robot = cat(1, bug2Path_robot, point);
        end
    end
    pose = rtde.movel(bug2Path_robot);
    poses = cat(1,poses,pose);
% Only one step
else
    point = [[bug2Path_world(size(bug2Path_world, 1), :), zaxis_high], (home_position(4:6))];
    pose = rtde.movej(point);
    poses = cat(1,poses,pose);
end

% Move down
goal_pt = [[bug2Path_world(size(bug2Path_world, 1), :), zaxis_zero], (home_position(4:6))];
pose = rtde.movej(goal_pt);
poses = cat(1,poses,pose);
% Release
disp("Release")
vacuum.release()

pose = rtde.movej(home_position);
rtde.drawPath(poses);

% End of program
rtde.close;
disp('Program Complete')

%% Functions
function path = getBug2Path(maze, startPos, goalPos)
    % Gat M path
    currentPos = startPos;
    Mpath = [currentPos];
    while ~isequal(currentPos, goalPos)
        % Check if the next position in the current direction is valid
        possibleDirections = [0, -1; 1, 0; 0, 1; -1, 0];  % Left, Down, Right, Up
        [~, bestIdx] = min(sum((currentPos + possibleDirections - goalPos).^2, 2));
        direction = possibleDirections(bestIdx, :);
        nextPos = currentPos + direction;
        Mpath = [Mpath; nextPos];
        currentPos = nextPos;
    end

    for i = 1:size(Mpath, 1)
        if maze(Mpath(i, 1), Mpath(i, 2)) == 0
            maze(Mpath(i, 1), Mpath(i, 2)) = 3;
        end
    end

    % Define obstacle edge
    for i = 2:size(maze, 1)-1
        for j = 2:size(maze, 2)-1
            if maze(i, j)==4
                if maze(i+1, j-1)==0 || maze(i+1, j-1)==3
                    maze(i+1, j-1)=2;
                end
                if maze(i+1, j)==0 || maze(i+1, j)==3
                    maze(i+1, j)=2;
                end
                if maze(i+1, j+1)==0 || maze(i+1, j+1)==3
                    maze(i+1, j+1)=2;
                end
                if maze(i-1, j-1)==0 || maze(i-1, j-1)==3
                    maze(i-1, j-1)=2;
                end
                if maze(i-1, j)==0 || maze(i-1, j)==3
                    maze(i-1, j)=2;
                end
                if maze(i-1, j+1)==0 || maze(i-1, j+1)==3
                    maze(i-1, j+1)=2;
                end
                if maze(i, j-1)==0 || maze(i, j-1)==3
                    maze(i, j-1)=2;
                end
                if maze(i, j+1)==0 || maze(i, j+1)==3
                    maze(i, j+1)=2;
                end
            end
        end
    end
    bug2_maze = maze;

    path = [];
    % Initialize the current position and direction
    currentPos = startPos;
    prePos = goalPos;
    path = [path; currentPos];
    direction = [0, -1];  % Initial direction: left
    findMpath = false;

    % Bug2 algorithm
    while ~isequal(currentPos, goalPos)

        possibleDirections = [0, -1; 1, 0; 0, 1; -1, 0];  % Left, Down, Right, Up
        possibleObstacles = [0, -1; 1, -1; 1, 0; 1, 1; 0, 1; -1, 1; -1, 0; -1, -1];  % 8 neighbours

        % Check if trapped
        trapped_count = 0;
        for i = 1:size(possibleDirections, 1)
            direction = possibleDirections(i, :);
            nextpossiblePos = currentPos + direction;
            if maze(nextpossiblePos(1), nextpossiblePos(2)) == 4 || maze(nextpossiblePos(1), nextpossiblePos(2)) == 1
                trapped_count = trapped_count + 1;
            else
                backPos = nextpossiblePos;
            end
        end
        if trapped_count > 2
            disp('Trapped, move back!')
            % Set currentPos to 4 as an obstacle
            maze(currentPos(1), currentPos(2)) = 4;
            prePos = currentPos;
            currentPos = backPos;
            path = [path; currentPos];
            continue
        end

        % Check if there is an adjacent Mpath
        findMpath = false;
        nextpossibleMpath = [];
        for i = 1:size(possibleDirections, 1)
            direction = possibleDirections(i, :);
            nextpossiblePos = currentPos + direction;
            if maze(nextpossiblePos(1), nextpossiblePos(2)) == 3 && ~isequal(prePos, nextpossiblePos)
                nextpossibleMpath = [nextpossibleMpath; nextpossiblePos];
                findMpath = true;
            end
        end

        % There is an adjacent Mpath
        if findMpath == true && size(nextpossibleMpath, 1) == 1
            disp('one adjacent Mpath')
            % Set currentPos to 0
            maze(currentPos(1), currentPos(2)) = 0;
            prePos = currentPos;
            currentPos = nextpossibleMpath;
            path = [path; currentPos];

        % There is more than one adjacent Mpath
        elseif findMpath == true && size(nextpossibleMpath, 1) > 1
            disp('multiple adjacent Mpath')
            % Get the path which has smaller distance to the goal
            minDist = size(maze, 1)+size(maze, 2);
            minIdx = 0;
            for i = 1:size(nextpossibleMpath, 1)
                dist = norm(nextpossibleMpath(i, :)-goalPos);
                if dist < minDist
                    minDist = dist;
                    minIdx = i;
                end
            end
            % Set currentPos to 0
            maze(currentPos(1), currentPos(2)) = 0;
            prePos = currentPos;
            currentPos = nextpossibleMpath(round(minIdx), :);
            path = [path; currentPos];

        % No adjacent Mpath, move counterclockwise around obstacle
        % If on edge, check where is the obstacle
        elseif maze(currentPos(1), currentPos(2)) == 2
            disp('on edge')
            nextpossibleDir = [];
            for i = 1:size(possibleDirections, 1)
                direction = possibleDirections(i, :);
                nextPos = currentPos + direction;
                if maze(nextPos(1), nextPos(2)) == 2 && ~isequal(prePos, nextPos)
                    nextpossibleDir = [nextpossibleDir; direction];
                end
            end
            
            % There is only one possible direction
            if size(nextpossibleDir, 1) == 1
                nextPos = currentPos + nextpossibleDir;
                prePos = currentPos;
                currentPos = nextPos;
                path = [path; currentPos];

            % There are more than one possible directions
            elseif ~isequal(nextpossibleDir, [])
                % Get the direction of the obstacle
                for i = 1:size(possibleObstacles, 1)
                    obstDirection = possibleObstacles(i, :);
                    obstPos = currentPos + obstDirection;
                    if maze(obstPos(1), obstPos(2)) == 4
                        obstPos;
                        break
                    end
                end
                % TODO: Compare the possible directions with the obstacle direction
                findDircetion = false;

%                 findDircetion
%                 nextpossibleDir
                % If all cross product < 0, choose the last nextpossibleDir
                if findDircetion == false
                    minDist = size(maze, 1)+size(maze, 2);
                    minIdx = 0;
                    for i = 1:size(nextpossibleDir, 1)
                        dist = norm(currentPos+nextpossibleDir(i, :)-goalPos);
                        if dist < minDist
                            minDist = dist;
                            minIdx = i;
                        end
                    end
                    nextPos = currentPos + nextpossibleDir(minIdx, :);
                    prePos = currentPos;
                    currentPos = nextPos;
                    path = [path; currentPos];
                end
            % There is no possible direction
            % Trapped, move to another empty(0) position
            else
                disp('Location 5, Trapped on edge')
                for i = 1:size(possibleDirections, 1)
                    direction = possibleDirections(i, :);
                    nextpossiblePos = currentPos + direction;
                    if maze(nextpossiblePos(1), nextpossiblePos(2)) == 0 && ~isequal(maze(nextpossiblePos(1), nextpossiblePos(2)), prePos)
                        backPos = nextpossiblePos;
                    end
                end
                maze(currentPos(1), currentPos(2)) = 4;
                prePos = currentPos;
                currentPos = backPos;
                path = [path; currentPos];
            end

        % if not on edge, move to edge
        else
            disp('not on edge')
            findDircetion = false;
            for i = 1:size(possibleDirections, 1)
                direction = possibleDirections(i, :);
                nextPos = currentPos + direction;
                if maze(nextPos(1), nextPos(2)) == 2 && ~isequal(prePos, nextPos)
                    findDircetion = true;
                    % Set currentPos to 0
                    maze(currentPos(1), currentPos(2)) = 0;
                    prePos = currentPos;
                    currentPos = nextPos;
                    path = [path; currentPos];
                    break
                end
            end
        end
    end
end