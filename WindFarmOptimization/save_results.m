function save_results(farmlayout,farmlayout_NA,i,folder)

save(sprintf('%s/farmlayout%d.mat',folder,i),"farmlayout")
save(sprintf('%s/farmlayoutNA%d.mat',folder,i),"farmlayout_NA")

end
