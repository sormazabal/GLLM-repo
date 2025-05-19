% markerList = {'ABCB1','ABCC1','ABCG2','ALDH1A1'};
%targetList = {'os'};
markerList = {'CD44','ALCAM','PROM1'};
for markeridx = 1:length(markerList)
    bioMarker = char(markerList(markeridx));
    disp(bioMarker) 
    run step5_nonstemness.m
    run step5_stemness.m
end



% for targetidx = 1:length(targetList)
%     pred_targ = char(targetList(targetidx));
%     disp(pred_targ)
%     for markeridx = 1:length(markerList) 
%         bioMarker = char(markerList(markeridx));
%         disp(bioMarker)
%         run step5_nonstemness.m
%         run step5_stemness.m
%     end
% end
