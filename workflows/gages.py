from stormhub.hydro.usgs.usgs import new_gage_collection, new_gage_catalog, add_ams_swe_to_gage_collection
from stormhub.hydro.utils import find_gages_in_watershed
from stormhub.hydro.usgs.calibration_events import add_calibration_events_to_collection
from stormhub.logger import initialize_logger


if __name__ == "__main__":

    initialize_logger()
    catalog_id = "watershed_USGS_Gage_Catalog"
    local_directory = "/path/for/created/stac"
    watershed = "path/of/watershed.geojson"
    swe_zarr_folder = "/workspaces/stormhub/data/0_source/swe/zarr"
    drainage_area_geojson_path = "/workspaces/stormhub/data/0_source/basins/gage_basins.geojson"

    gage_catalog = new_gage_catalog(
        catalog_id,
        local_directory=local_directory,
        catalog_description="watershed USGS Gage Catalog",
    )

    # gage_numbers = ["12105900", "12167000"] # test gages
    gage_numbers = find_gages_in_watershed(watershed, 15)
    collection = new_gage_collection(gage_catalog, gage_numbers, local_directory)

    # Optionally add average SWE data to each gage item for each AMS event
    add_ams_swe_to_gage_collection(
        gage_collection = gage_catalog,
        drainage_area_geojson_path = drainage_area_geojson_path,
        swe_zarr_path = swe_zarr_folder,
        swe_threshold_mm = 10)
