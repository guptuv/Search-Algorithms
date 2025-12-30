clear; clc; close all;

%% ================= PARAMETERS =================
map.x = [0 50];
map.y = [0 50];
map.z = [0 20];

start = [5 5 5];
goal  = [45 45 10];

step_size   = 3.0;
goal_thresh = 2.0;
goal_bias   = 0.4;
max_iter    = 1500;

%% ================= FIGURE =================
figure; hold on; grid on; axis equal;
xlim(map.x); ylim(map.y); zlim(map.z);
xlabel('X'); ylabel('Y'); zlabel('Z');
view(3);

scatter3(start(1), start(2), start(3), 120, 'g', 'filled');
scatter3(goal(1),  goal(2),  goal(3),  120, 'r', 'filled');

%% ================= TREE INIT =================
tree(1).pos = start;
tree(1).parent = 0;

%% ================= RRT LOOP =================
for i = 2:max_iter

    % -------- Sample --------
    if rand < goal_bias
        q_rand = goal;
    else
        q_rand = [ ...
            rand_range(map.x), ...
            rand_range(map.y), ...
            rand_range(map.z) ];
    end

    % -------- Nearest node --------
    idx_near = nearest_node(tree, q_rand);
    q_near = tree(idx_near).pos;

    % -------- Steer --------
    dir = q_rand - q_near;
    if norm(dir) == 0
        continue;
    end
    q_new = q_near + step_size * dir / norm(dir);

    % -------- Add node --------
    tree(i).pos = q_new;
    tree(i).parent = idx_near;

    plot3([q_near(1) q_new(1)], ...
          [q_near(2) q_new(2)], ...
          [q_near(3) q_new(3)], 'b');

    drawnow limitrate;

    % -------- Goal check --------
    if norm(q_new - goal) < goal_thresh
        tree(i+1).pos = goal;
        tree(i+1).parent = i;

        plot3([q_new(1) goal(1)], ...
              [q_new(2) goal(2)], ...
              [q_new(3) goal(3)], ...
              'r', 'LineWidth', 2);

        disp('GOAL REACHED');
        break;
    end
end

%% ================= PATH EXTRACTION =================
path = [];
idx = length(tree);

while idx ~= 0
    path = [tree(idx).pos; path];
    idx = tree(idx).parent;
end

plot3(path(:,1), path(:,2), path(:,3), ...
      'm', 'LineWidth', 3);

title('3D RRT Path');

%% ================= FUNCTIONS =================
function r = rand_range(b)
    r = b(1) + rand * (b(2) - b(1));
end

function idx = nearest_node(tree, q)
    dmin = inf;
    idx = 1;
    for k = 1:length(tree)
        d = norm(tree(k).pos - q);
        if d < dmin
            dmin = d;
            idx = k;
        end
    end
end
