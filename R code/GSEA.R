
library(ReactomePA)
library(tidyverse)
library(data.table)
library(org.Hs.eg.db)
library(clusterProfiler)
library(biomaRt)
library(enrichplot)  

KEGG_gseresult <- gseKEGG(geneList,  
                          organism = 'hsa',
                          nPerm = 1000, 
                          minGSSize = 10, 
                          maxGSSize = 1000,
                          pvalueCutoff=0.25,
                          pAdjustMethod= "BH")


gseaplot2(
  KEGG_gseresult,
  id,
  pvalue_table = FALSE,
  ES_geom = "line"
)
write.table(KEGG_gseresult, file ="KEGG_gseresult.csv", sep =",", row.names =FALSE)
