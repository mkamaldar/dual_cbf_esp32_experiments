function plot_closed_loop(datafile)
%PLOT_CLOSED_LOOP  Closed-loop figure (Fig. 2) of "Extracting exact Lie
%   derivatives without backpropagation: A dual compiler for neural control
%   barrier functions".
%
%   plot_closed_loop reads closed_loop_data.mat, which 03_run_closed_loop.py
%   writes next to closed_loop.pdf, and draws the three panels
%       (a) kinematic bicycle      : look-ahead lateral position (Experiment 1)
%       (b) Van der Pol oscillator : phase portrait              (Experiment 2)
%       (c) inverted pendulum      : angle                       (Experiment 3)
%   with tight_subplot.  It writes closed_loop.pdf (vector) and
%   closed_loop.eps.
%
%   plot_closed_loop(datafile) reads the indicated MAT-file instead.
%
%   Requires tight_subplot.m on the path.
%
%   The simulations are NOT repeated here.  They call the compiled C++
%   headers and are run by 03_run_closed_loop.py; this script only plots
%   the stored trajectories, so the figure and sim_results.txt come from
%   the same run.
%
%   All colors, line styles, line widths, labels, and sizes are collected
%   in the block "CUSTOMIZATION" below.

if nargin < 1, datafile = 'closed_loop_data.mat'; end
D = load(datafile);

%% ======================== CUSTOMIZATION ===============================
% --- curves: one struct per curve (label, color, line style, line width)
C.ideal     = struct('label','filtered, ideal',     'color',[1 .84 0], 'style','-',  'width',2);
C.perturbed = struct('label','filtered, perturbed', 'color',[0 .44 1], 'style','-.', 'width',1.2);
C.nominal   = struct('label','nominal',             'color',[0 0 0], 'style','--', 'width',1.0);
C.boundary  = struct('label','zero level set', 'color',[0.78 0.06 0.18], 'style','-', 'width',1.0);
% --- shaded set on which the barrier is nonnegative
safeColor = [0.90 0.93 0.96];
safeLabel = 'zero superlevel set';

% --- drawing order of the trajectories (the first is drawn at the bottom;
%     a patterned curve must be drawn after a solid curve that it overlaps)
order = {'ideal','perturbed','nominal'};

% --- marker of the initial state in the phase portrait
x0Color = [0.13 0.13 0.13];   x0Size = 3;

% --- figure and fonts
figW = 8.2;   figH = 2.35;                 % inches (full text width)
fontName = 'times';   fontSize = 9;   legendFontSize = 8;
showGrid = true;

% --- layout passed to tight_subplot(1,3,gap,[lower upper],[left right])
gap    = [0 0.06];                        % [vertical horizontal] gap
margH  = [0.15 0.20];                      % [lower upper]; upper holds titles and legend
margW  = [0.050 0.012];                    % [left right]; left holds the axis label of panel (a)

% --- axis limits, labels, and titles
B.ylim = [-9 9];
B.xlabel = '$t$ (s)';     B.ylabel = '$\bar p_y+\tau_{\rm h}\bar v\sin\bar\psi$ (m)';   B.title = '(a) Bicycle';
V.xlim = [-4 4];      V.ylim = [-3.5 3.5];
V.xlabel = '$x_{(1)}$';   V.ylabel = '$x_{(2)}$';   V.title = '(b) Van der Pol';
P.ylim = [-0.2 1.3];
P.xlabel = '$t$ (s)';     P.ylabel = '$\phi$ (rad)';   P.title = '(c) Pendulum';

% --- output file name (without extension)
fname = 'closed_loop';
%% ======================================================================

set(0,'defaultAxesFontName',fontName, ...
      'defaultTextFontName',fontName, ...
      'DefaultAxesFontSize',fontSize, ...
      'DefaultTextFontSize',fontSize);

h = figure(1); clf;
set(h,'Color','w','Units','inches');
set(h,'Position',[.25 .25 figW figH]);
set(h,'PaperUnits','inches','PaperPosition',[.25 .25 figW figH]);

ha = tight_subplot(1,3,gap,margH,margW);
axB = ancestor(ha(1),'axes');  axV = ancestor(ha(2),'axes');  axP = ancestor(ha(3),'axes');

% ------------------------------------------------ (a) bicycle
axes(axB); hold(axB,'on'); box(axB,'on');
tB = double(D.bic_t(:)');  bB = double(D.bic_bound);  TB = double(D.bic_T);
band(axB,[0 TB],bB,safeColor,C.boundary);
T.ideal = D.bic_ideal;  T.perturbed = D.bic_perturbed;  T.nominal = D.bic_nominal;
for j = 1:numel(order)
    c = C.(order{j});
    plot(axB,tB,double(T.(order{j})(:)'),c.style,'Color',c.color,'LineWidth',c.width);
end
finish_axes(axB,[0 TB],B.ylim,B.xlabel,B.ylabel,B.title,showGrid,fontSize);

% ------------------------------------------------ (b) Van der Pol
axes(axV); hold(axV,'on'); box(axV,'on');
g1 = double(D.vdp_g1(:)');  g2 = double(D.vdp_g2(:)');  H = double(D.vdp_h);
% region {h_theta >= 0}: contourf fills only the region above the level 0
contourf(axV,g1,g2,H,[0 0],'LineStyle','none','HandleVisibility','off');
colormap(axV,safeColor);
contour(axV,g1,g2,H,[0 0],'LineColor',C.boundary.color, ...
        'LineStyle',C.boundary.style,'LineWidth',C.boundary.width, ...
        'HandleVisibility','off');
T.ideal = D.vdp_ideal;  T.perturbed = D.vdp_perturbed;  T.nominal = D.vdp_nominal;
for j = 1:numel(order)
    c = C.(order{j});  X = T.(order{j});
    plot(axV,X(:,1),X(:,2),c.style,'Color',c.color,'LineWidth',c.width);
end
plot(axV,D.vdp_x0(1),D.vdp_x0(2),'o','MarkerSize',x0Size, ...
     'MarkerFaceColor',x0Color,'MarkerEdgeColor',x0Color);
finish_axes(axV,V.xlim,V.ylim,V.xlabel,V.ylabel,V.title,showGrid,fontSize);

% ------------------------------------------------ (c) pendulum
axes(axP); hold(axP,'on'); box(axP,'on');
bP = double(D.pen_bound);  TP = double(D.pen_T);
band(axP,[0 TP],bP,safeColor,C.boundary);
tt.ideal = D.pen_t_ideal;  tt.perturbed = D.pen_t_perturbed;  tt.nominal = D.pen_t_nominal;
T.ideal = D.pen_ideal;     T.perturbed = D.pen_perturbed;     T.nominal = D.pen_nominal;
for j = 1:numel(order)
    c = C.(order{j});
    plot(axP,double(tt.(order{j})(:)'),double(T.(order{j})(:)'),c.style, ...
         'Color',c.color,'LineWidth',c.width);
end
finish_axes(axP,[0 TP],P.ylim,P.xlabel,P.ylabel,P.title,showGrid,fontSize);

% ------------------------------------------------ legend (one row, top)
% Invisible proxy objects, so that the legend does not depend on the
% drawing order of the panels.
hI = plot(axB,nan,nan,C.ideal.style,    'Color',C.ideal.color,    'LineWidth',C.ideal.width);
hP = plot(axB,nan,nan,C.perturbed.style,'Color',C.perturbed.color,'LineWidth',C.perturbed.width);
hN = plot(axB,nan,nan,C.nominal.style,  'Color',C.nominal.color,  'LineWidth',C.nominal.width);
hS = patch(axB,nan(1,3),nan(1,3),safeColor,'EdgeColor','none');
hB = plot(axB,nan,nan,C.boundary.style, 'Color',C.boundary.color, 'LineWidth',C.boundary.width);
hl = [hI hP hN hS hB];
labels = {C.ideal.label,C.perturbed.label,C.nominal.label,safeLabel,C.boundary.label};
lg = legend(axB,hl,labels,'Orientation','horizontal','Box','off', ...
            'FontSize',legendFontSize,'Interpreter','latex');
set(lg,'Units','normalized');
p = get(lg,'Position');
set(lg,'Position',[0.5-p(3)/2, 1-p(4)-0.005, p(3), p(4)]);   % centered at the top

% ------------------------------------------------ export
set(h,'PaperPositionMode','auto');
if exist('exportgraphics','file') == 2 || exist('exportgraphics','builtin') == 5
    exportgraphics(h,[fname '.pdf'],'ContentType','vector');
else
    set(h,'PaperUnits','inches','PaperSize',[figW figH],'PaperPosition',[0 0 figW figH]);
    print(h,fname,'-dpdf','-painters');
end
print(h,fname,'-depsc','-painters');
end

%% ---------------------------------------------------------------------
function band(ax,tlim,b,safeColor,cb)
%BAND  Shaded band |y| <= b with its two boundary lines.
patch(ax,[tlim(1) tlim(2) tlim(2) tlim(1)],[-b -b b b],safeColor, ...
      'EdgeColor','none','HandleVisibility','off');
plot(ax,tlim,[ b  b],cb.style,'Color',cb.color,'LineWidth',cb.width,'HandleVisibility','off');
plot(ax,tlim,[-b -b],cb.style,'Color',cb.color,'LineWidth',cb.width,'HandleVisibility','off');
end

%% ---------------------------------------------------------------------
function finish_axes(ax,xl,yl,xlab,ylab,ttl,showGrid,fontSize)
%FINISH_AXES  Limits, labels, title, ticks, and grid of one panel.
xlim(ax,xl);  ylim(ax,yl);
set(ax,'Layer','top','XTickLabelMode','auto','YTickLabelMode','auto', ...
       'FontSize',fontSize-1);
xlabel(ax,xlab,'Interpreter','latex','FontSize',fontSize);
ylabel(ax,ylab,'Interpreter','latex','FontSize',fontSize);
title(ax,ttl,'FontWeight','normal','FontSize',fontSize);
if showGrid
    grid(ax,'on');  set(ax,'GridLineStyle',':','GridAlpha',0.35);
end
end
