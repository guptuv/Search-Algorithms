clear; clc; close all;

%% ================= MAP =================
map.x = [0 50];
map.y = [0 50];
map.z = [0 50];

start = [5 5 5];
goal  = [45 45 45];

step_size   = 1.0;
goal_thresh = 2.0;
goal_bias   = 0.3;
max_iter    = 2500;
gamma_rrt   = 10;

%% ================= OBSTACLES =================
obstacles = [
    10 18 10 18   0 12;
    22 30 22 30   0 45;
    35 43  8 18   0 35;
    15 25 32 42   0 30;
    20 30  5 45  20 30;
];

%% ================= FIGURE =================
figure; hold on; grid on; axis equal;
xlim(map.x); ylim(map.y); zlim(map.z);
xlabel('X'); ylabel('Y'); zlabel('Z');
view(3);

scatter3(start(1),start(2),start(3),120,'g','filled');
scatter3(goal(1),goal(2),goal(3),120,'r','filled');
draw_obstacles(obstacles);

%% ================= TREE INIT =================
tree(1).pos = start;
tree(1).parent = 0;
tree(1).cost = 0;
node_count = 1;

best_goal_cost = inf;
best_goal_idx  = -1;

%% ================= RRT* =================
for iter = 1:max_iter

    % Sample
    if rand < goal_bias
        q_rand = goal;
    else
        q_rand = [ ...
            rand_range(map.x), ...
            rand_range(map.y), ...
            rand_range(map.z) ];
    end

    % Nearest
    idx_near = nearest_node(tree, node_count, q_rand);
    q_near = tree(idx_near).pos;

    % Steer
    dir = q_rand - q_near;
    if norm(dir) == 0
        continue;
    end
    q_new = q_near + step_size * dir / norm(dir);

    if collision_segment(q_near, q_new, obstacles)
        continue;
    end

    % Neighborhood
    r = min(gamma_rrt * (log(node_count)/node_count)^(1/3), step_size*5);
    near_ids = near_nodes(tree, node_count, q_new, r);

    % Best parent
    min_cost = tree(idx_near).cost + norm(q_new - q_near);
    best_parent = idx_near;

    for i = near_ids
        c = tree(i).cost + norm(q_new - tree(i).pos);
        if c < min_cost && ~collision_segment(tree(i).pos, q_new, obstacles)
            min_cost = c;
            best_parent = i;
        end
    end

    % Add node
    node_count = node_count + 1;
    tree(node_count).pos = q_new;
    tree(node_count).parent = best_parent;
    tree(node_count).cost = min_cost;

    plot3([tree(best_parent).pos(1) q_new(1)], ...
          [tree(best_parent).pos(2) q_new(2)], ...
          [tree(best_parent).pos(3) q_new(3)], 'b');

    % Rewire
    for i = near_ids
        new_cost = tree(node_count).cost + norm(tree(i).pos - q_new);
        if new_cost < tree(i).cost && ...
           ~collision_segment(tree(i).pos, q_new, obstacles)
            tree(i).parent = node_count;
            tree(i).cost = new_cost;
        end
    end

    % Goal check
    if norm(q_new - goal) < goal_thresh && ...
       ~collision_segment(q_new, goal, obstacles)

        total_cost = tree(node_count).cost + norm(q_new - goal);
        if total_cost < best_goal_cost
            best_goal_cost = total_cost;
            best_goal_idx  = node_count;
        end
    end
end

%% ================= PATH EXTRACTION =================
path = goal;
idx = best_goal_idx;
while idx ~= 0
    path = [tree(idx).pos; path];
    idx = tree(idx).parent;
end

plot3(path(:,1),path(:,2),path(:,3),'m','LineWidth',3);

%% ================= SMOOTHING =================
smooth = shortcut_smoothing(path, obstacles, 600);
plot3(smooth(:,1), smooth(:,2), smooth(:,3), 'k', 'LineWidth', 4);

%% ================= PX4 WAYPOINTS =================
waypoints = downsample_path(smooth, 2.0);
waypoints(:,3) = smoothdata(waypoints(:,3),'movmean',3);

plot3(waypoints(:,1),waypoints(:,2),waypoints(:,3), ...
      'ko','MarkerSize',6,'LineWidth',2);

export_px4_waypoints(waypoints, 'rrt_star_px4.waypoints');

title('RRT* → Smoothed → PX4 Waypoints');

%% ================= FUNCTIONS =================
function r = rand_range(b)
    r = b(1) + rand * (b(2) - b(1));
end

function idx = nearest_node(tree, n, q)
    dmin = inf; idx = 1;
    for k = 1:n
        d = norm(tree(k).pos - q);
        if d < dmin
            dmin = d; idx = k;
        end
    end
end

function ids = near_nodes(tree, n, q, r)
    ids = [];
    for k = 1:n
        if norm(tree(k).pos - q) <= r
            ids(end+1) = k; %#ok<AGROW>
        end
    end
end

function hit = collision_segment(p1, p2, obs)
    hit = false;
    for i = 1:size(obs,1)
        if segment_box_intersect(p1, p2, obs(i,:))
            hit = true; return;
        end
    end
end

function inside = segment_box_intersect(p1, p2, box)
    inside = false;
    for t = linspace(0,1,40)
        p = p1 + t*(p2-p1);
        if p(1)>=box(1) && p(1)<=box(2) && ...
           p(2)>=box(3) && p(2)<=box(4) && ...
           p(3)>=box(5) && p(3)<=box(6)
            inside = true; return;
        end
    end
end

function smooth_path = shortcut_smoothing(path, obstacles, max_iter)
    smooth_path = path;
    for k = 1:max_iter
        if size(smooth_path,1) <= 2, break; end
        i = randi([1 size(smooth_path,1)-1]);
        j = randi([i+1 size(smooth_path,1)]);
        if ~collision_segment(smooth_path(i,:), smooth_path(j,:), obstacles)
            smooth_path = [smooth_path(1:i,:); smooth_path(j:end,:)];
        end
    end
end

function wp = downsample_path(path, min_dist)
    wp = path(1,:);
    last = path(1,:);
    for i = 2:size(path,1)
        if norm(path(i,:) - last) >= min_dist
            wp = [wp; path(i,:)]; %#ok<AGROW>
            last = path(i,:);
        end
    end
    if norm(wp(end,:) - path(end,:)) > 1e-3
        wp = [wp; path(end,:)];
    end
end

function export_px4_waypoints(wp, filename)
    fid = fopen(filename,'w');
    fprintf(fid,'QGC WPL 110\n');
    for i = 1:size(wp,1)
        fprintf(fid,'%d\t0\t3\t16\t0\t0\t0\t0\t%.6f\t%.6f\t%.6f\t1\n', ...
            i-1, wp(i,1), wp(i,2), wp(i,3));
    end
    fclose(fid);
end

function draw_obstacles(obs)
    for i = 1:size(obs,1)
        b = obs(i,:);
        [X,Y,Z] = meshgrid([b(1) b(2)], [b(3) b(4)], [b(5) b(6)]);
        X = X(:); Y = Y(:); Z = Z(:);
        K = convhull(X,Y,Z);
        trisurf(K,X,Y,Z,'FaceAlpha',0.3,...
            'FaceColor',[0.8 0.2 0.2],'EdgeColor','none');
    end
end
