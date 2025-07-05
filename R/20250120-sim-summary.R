library(ggplot2)
library(stringr)

date_ <- "20250627" 
date_ <- format(lubridate::today(), "%Y%m%d")

t_test_file <- here::here("sim-output", str_glue("{date_}-table01a-t-test.xlsx"))  

ancova_file <- here::here("sim-output", str_glue("{date_}-table02a-ancova.xlsx"))

mmrm_type <- "a"
mmrm_file   <- here::here("sim-output", str_glue("{date_}-table03{mmrm_type}-mmrm.xlsx"))

nlmem_file  <- here::here("sim-output", str_glue("{date_}-table04a-nlmem.xlsx"))

t_test <- readxl::read_xlsx(t_test_file)
ancova <- readxl::read_xlsx(ancova_file)
mmrm <- readxl::read_xlsx(mmrm_file)

nlmem1  <- readxl::read_xlsx(nlmem_file) |> 
  dplyr::select(-AUC, -`ηα`) |> 
  dplyr::rename(`Randomization-based` = CHFBL)
nlmem2  <- readxl::read_xlsx(nlmem_file) |> 
  dplyr::select(-AUC, -CHFBL) |> 
  dplyr::rename(`Randomization-based` = `ηα`)


nlmem <- nlmem1
case <- 2
if(case == 2) {
  nlmem <- nlmem2
}
nlmem

test_df <- purrr::map2_df(
  list(t_test, ancova, mmrm, nlmem),
  c("t-test", "ANCOVA", "MMRM", "NLMEM"), ~ {
    tbl <- .x
    lbl <- .y
    tbl |> 
      dplyr::mutate(
        flag1 = (scenario-1) %/% 4 + 1, 
        flag2 = (scenario-1) %/% 2 + 1, .before = 1
      ) |> 
      dplyr::group_by(flag1) |> 
      dplyr::group_nest() |> 
      dplyr::mutate(
        data = purrr::map(data, ~ {
          trteff <- .x$`Treatment effect`[1]
          .x |> 
            dplyr::mutate(
              `Treatment effect` = dplyr::if_else(
                is.na(`Treatment effect`), trteff, `Treatment effect`
              )
            )
        })
      ) |> 
      tidyr::unnest("data") |> 
      dplyr::group_by(flag2) |> 
      dplyr::group_nest() |> 
      dplyr::mutate(
        data = purrr::map(data, ~ {
          lintrend <- .x$`Linear trend`[1]
          .x |> 
            dplyr::mutate(
              `Linear trend` = dplyr::if_else(
                is.na(`Linear trend`), lintrend, `Linear trend`
              )
            )
        })
      ) |> 
      tidyr::unnest("data") |> 
      tibble::add_column(analysis = lbl)
  }) |> 
  dplyr::select(-flag1, -flag2) |> 
  tidyr::pivot_longer(
    dplyr::ends_with("-based"),
    names_to = "approach",
    values_to = "error"
  ) |> 
  dplyr::mutate(
    ylabel = factor(
      stringr::str_glue("{Randomization} ({analysis})"),
      levels = c(
        c("TBD (NLMEM)", "TBD (ANCOVA)", "TBD (MMRM)", "TBD (t-test)"),
        c("Rand (NLMEM)", "Rand (ANCOVA)", "Rand (MMRM)", "Rand (t-test)")
      )
    ),
    analysis_approach = factor(
      stringr::str_glue("{analysis}, {approach}"),
      levels = c(
        c("t-test, Population-based", "t-test, Randomization-based"), 
        c("ANCOVA, Population-based", "ANCOVA, Randomization-based"),
        c("MMRM,   Population-based", "MMRM, Randomization-based"),
        c("NLMEM,  Population-based", "NLMEM, Randomization-based")
      )
    ),
    `Linear trend` = factor(
      `Linear trend`,
      levels = c("Yes", "No"),
      labels = c("Linear trend=Yes", "Linear trend=No")
    ),
    DE = stringr::str_glue("DE={`Treatment effect`}"),
    error_txt = stringr::str_glue("{round(error, 3)*100}")
  ) 
test_df

test_df |> 
  dplyr::mutate(
    SE = dplyr::if_else(`Treatment effect` == 0, sqrt(error * (1-error) / 2500), 0),
    analysis = str_glue("{analysis} ({Randomization})"),
    analysis = factor(
      analysis,
      levels = c(
        "NLMEM (Rand)",  "NLMEM (TBD)",
        "MMRM (Rand)", "MMRM (TBD)",
        "ANCOVA (Rand)", "ANCOVA (TBD)",
        "t-test (Rand)", "t-test (TBD)" 
      )
    ),
    #error_y = error + dplyr::if_else(`Treatment effect` == 0, 0.0025, 0.05),
    error_y = error + dplyr::if_else(`Treatment effect` == 0, 0.007, 0.05),
    nominal = dplyr::if_else(`Treatment effect` == 0, 0.05, -0.1),
    nominal1 = dplyr::if_else(`Treatment effect` == 0, error, -0.1)
  ) |> 
  ggplot()+
  geom_col(
    aes(error, analysis, fill = approach), 
    color = "white",
    position = "dodge"
  )+
  geom_errorbar(
    aes(x = nominal1, y = analysis, xmin = nominal1 - 1.96*SE, xmax = nominal1 + 1.96*SE),
    position = position_dodge2(width = 0.9, padding = 0.8),
    width = 0.9,
    color = "black",
    linewidth = 0.5
  )+
  geom_text(
    aes(error_y, analysis, label = error_txt), 
    position = position_dodge2(width = 0.9, padding = 0.5, reverse = TRUE),
    size = 3.5,
    fontface = "bold", angle = 30
  )+
  geom_vline(aes(xintercept = nominal), linetype = "dashed", linewidth = 0.75)+
  xlab("type I error rate / power (%)")+
  ylab(NULL)+
  facet_grid(`Linear trend` ~ DE, scales="free_x")+
  ggh4x::facetted_pos_scales(
    x = list(
      DE == "DE=0" ~ scale_x_continuous(
        limits = c(0, 0.15),
        breaks = seq(0, 0.15, by = 0.05),
        labels = str_glue("{100*seq(0, 0.15, by = 0.05)}")
      ),
      TRUE ~ scale_x_continuous(
        limits = c(0, 1.1),
        breaks = seq(0, 1, by = 0.05),
        labels = str_glue("{100*seq(0, 1, by = 0.05)}")
      )
    )
  )+
  theme_bw(base_size = 14)+
  theme(
    legend.position = "bottom",
    strip.text = element_text(face = "bold"),
    strip.background = element_blank(),
    axis.text.x = element_text(size = 8, face = "bold", angle = 30),
    axis.title.x = element_text(size = 14, face = "bold"),
    axis.text.y = element_text(face = "bold"),
    axis.ticks.y = element_blank(),
    axis.ticks.x = element_blank(),
    legend.text = element_text(face = "bold"),
    legend.title = element_blank(),
    panel.grid.minor.x = element_blank()
  ) 

ggsave(
  filename = here::here(str_glue("sim-output/{date_}-analysis-output{mmrm_type}.jpeg")),
  width = 16,
  height = 9,
  units = "in",
  dpi = 300
)