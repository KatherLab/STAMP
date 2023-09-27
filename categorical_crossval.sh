true=ABOVE
for full_dir in /home/omarelnahhas/omars_hdd/omar/omar/immunoproject/new/T*CRC*
do
	for target in "TIL Regional Fraction"
	do
		dir=${full_dir##*/}
		echo "Training non-site-based splitting median classification model with ${dir} on ${target} median"
		python ./modeling/modeling.py \
			--clini_table /home/omarelnahhas/omars_hdd/omar/omar/immunoproject_project/immune_class.xlsx \
			--slide_csv ${full_dir}/*_SLIDE.csv  \
			--feature_dir ${full_dir}/e2e-xiyue-wang-macenko \
			--output_path /home/omarelnahhas/omars_hdd/omar/omar/STAMP/test/${dir} \
			--target_label "${target} median" \
			--categories "ABOVE" "BELOW" \
			--n_splits 5
	done
done
