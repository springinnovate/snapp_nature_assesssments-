from osgeo import gdal
import pandas as pd

gdal.UseExceptions()
HUC_TABLE_PATH = (
    r"D:\repositories\zonal_stats_toolkit\snappdata\huc_summary_forSNAPP_v2.csv"
)

HUC_VECTOR_PATH = r"D:\repositories\zonal_stats_toolkit\snappdata\WBD_National_GDB\WBD_National_GDB.gdb"
WETLANDS_MASK_RASTER_PATH = r"D:\repositories\zonal_stats_toolkit\snappdata\reclassified_Annual_NLCD_wetlands_reclass.tif"
ANNUAL_VALUE_FIELD = "marginal_annualvalue_2020"
MARGINAL_NPV_FIELD = "marginal_npv_2020"
HUC_FIELD = "huc12"
WETLAND_AREA_FIELD = "wetland_area_ha_2023"


def main():
    huc_table = pd.read_csv(HUC_TABLE_PATH)

    total_marginal_annual_value_2020 = sum(
        huc_table[WETLAND_AREA_FIELD] * huc_table[ANNUAL_VALUE_FIELD]
    )
    print(f"total_marginal_annual_value_2020: ${total_marginal_annual_value_2020:,.2f}")

    total_marginal_npv_value_2020 = sum(
        huc_table[WETLAND_AREA_FIELD] * huc_table[MARGINAL_NPV_FIELD]
    )
    print(f"total_marginal_npv_value_2020: ${total_marginal_npv_value_2020:,.2f}")

    huc_vector = gdal.OpenEx(HUC_VECTOR_PATH, gdal.OF_VECTOR)
    huc_layer = huc_vector.GetLayer()
    huc_set = set(huc_feature.GetField(HUC_FIELD) for huc_feature in huc_layer)
    print(len(huc_set))
    # for huc_feature in huc_layer:
    #     print(huc_feature.GetField(HUC_FIELD))
    # print(huc_layer)


if __name__ == "__main__":
    main()
