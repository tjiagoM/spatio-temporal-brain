library("fsbrain")

# Rscript --no-save --no-restore --verbose diffpool_plot_brains.R

subjects_dir <- fsbrain::get_optional_data_filepath("subjects_dir")
atlas <- 'aparc'  # Desikan atlas

sweep_names <- c('100_n_e_diffpool', '100_n_diffpool')

for (sweep_name in sweep_names)
{
  granularities <- c(4, 8, 12)
  if (sweep_name == '100_n_diffpool')
  {
    granularities <- c(4)
  }

  for (granularity_id in granularities)
  {
    for (group_id in c('female', 'male', 'total'))
    {
      # Left Hemisphere
      lh_map <- read.csv(paste('results/dp_clust_', granularity_id, '_', sweep_name, '_l_', group_id, '.csv', sep=''))
      lh_list <- lh_map$cluster
      names(lh_list) <-  lh_map$label

      # Right Hemisphere
      rh_map <- read.csv(paste('results/dp_clust_', granularity_id, '_', sweep_name, '_r_', group_id, '.csv', sep=''))
      rh_list <- rh_map$cluster
      names(rh_list) <-  rh_map$label

      saving_path <- paste('figures/dp_interp_', granularity_id, '_', sweep_name, '_', group_id, '.png', sep='')

      vis.region.values.on.subject(subjects_dir,
                'subject1',
                atlas,
                lh_list,
                rh_list,
                views = c('t4'),
                #draw_colorbar=TRUE,
                makecmap_options = list('colFn'=colorRampPalette(RColorBrewer::brewer.pal(n=granularity_id, name='Paired'))),
                rglactions = list("snapshot_png"=saving_path));
    }
  }
}