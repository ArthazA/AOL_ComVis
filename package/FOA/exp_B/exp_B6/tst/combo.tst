set Ldir_foa = "/proj/mgn-sar2/data_package/package/FOA/exp_B/exp_B6/trn/"
set Ldir_out = "/proj/mgn-sar2/data_package/package/FOA/exp_B/exp_B6/tst/"

#input images
        set trn_numimages = 6
        set trn_imagedir = "/proj/mgn-sar2/data_package/package/Images/"
        set trn_images = ( \
                img37 \
                img38 \
                img39 \
                img40 \
                img41 \
                img42 \
        )

#input truths
        set trn_truthdir = "/proj/mgn-sar2/data_package/package/GroundTruths/"
        set trn_truths = ( \
                img37.jtri \
                img38.jtri \
                img39.jtri \
                img40.jtri \
                img41.jtri \
                img42.jtri \
        )

# input training and testing image parameters
	set fullressize = 1024  # full resolution size (1024 x 1024)
        set NR0 = $fullressize  # image rows
        set NC0 = $fullressize  # image cols

# system parameters (used throughout learning and production)
# FOA is a matched filter
	set foa_threshold = 0.35  # FOA cut off threshold for selecting
				  # candidate locations (current best
				  # system uses 0.35)
	set patchsize = 30	# size of rectangular filter patch 
				# before spoiling (current best system
				# uses 30)
	set spoilfactor = 2	# spoil size (current best is 2)
	# these scripts cannot do division or multiplication so the user
	# must enter the spoilpatchsize (which = patchsize/spoilfactor)
	set spoilpatchsize = 15	# patch size after spoiling
	set remove_resps = 1	# 1=remove response images to save space

# scoring parameters
        set llbl = 1            # Minimum label in ground truth files
        set ulbl = 4            # Maximum label in ground truth files

	set border_thr = 15     # objects closer than this distance from
				# the edge are not used in scoring.
				# this should normally be set to half the
				# width of the patchsize.
	set cluster_thr = 4     # mcb says this should be 4.
				# two hits within this distance from each
				# other are attributed to the same object
				# and clustered together.
	set scoring_thr = 13	# mcb used 13 for this.
				# a detection must be at least this close
				# to the gnd truth location to be counted
				# as a hit (a true volcano).
				# if set to -1, then the gnf truth radii 
				# are used as the scoring thr.  min radii
				# allowed is rmin and max is rmax, both are
				# set in the sys.pars file.  

#-----------------------------------------------------------------
# matched filter
	set filterdir = $Ldir_foa
	set filter = "matched_filter" # name of matched filter

# similarity_xy parameters
# Note that these three parameters have been constant in all experiments
# to date.  If we start to want to change them regularly then they should
# be moved from this sys.pars file to the user.pars file.
	set inorm_similarity = 1  # 1=normalize image patch, 0=Not
	set fnorm_similarity = 1  # 1=normalize filter, 0=Not
	set ftol_similarity = 0.95  # filter approximation tolerance

# scoring parameters
# These are parameters to the scored_list2a program.  We've been using
# these default values all along.  If we start to want to use other
# values for them regularly then we should move these parameters to the
# user.pars file.
	set fa_label = 0   # label to give dets that don't match gnd truth
	set fa_color = "red"     # color of fas

# scoring threshold parameter value is set in the user.pars file.  If it
# is set to -1 then the user can also set rmin and rmax - the min and max
# radii of the ground truth regions allowed to be used as thresholds. If
# we want to change these parameters often, then they should be moved to
# the user.pars file.  Padhraic chose these default rmin and rmax.
	set rmin = 5
	set rmax = 15

# training parameters
	set trn_detectdir = $Ldir_out



