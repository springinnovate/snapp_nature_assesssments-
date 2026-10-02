@echo off
setlocal
set "COUNTY_MOVE_SCRIPT=%~f0"
set "COUNTY_MOVE_OPTION=%~1"
powershell.exe -NoLogo -NoProfile -Command "$text = Get-Content -LiteralPath $env:COUNTY_MOVE_SCRIPT -Raw; & ([scriptblock]::Create(($text -split '(?m)^# POWERSHELL_START\r?$', 2)[1]))"
set "result=%ERRORLEVEL%"
if not defined COUNTY_MOVE_OPTION pause
exit /b %result%
# POWERSHELL_START
# Run normally to move files, or use: move_county_inputs.cmd --dry-run
# Keep this script in the repository root. No analysis is performed.
$ErrorActionPreference = 'Stop'
if ($env:COUNTY_MOVE_OPTION -and $env:COUNTY_MOVE_OPTION -ne '--dry-run') {
    Write-Host 'Usage: move_county_inputs.cmd [--dry-run]'
    exit 1
}
$dryRun = $env:COUNTY_MOVE_OPTION -eq '--dry-run'
$downloads = 'C:\Users\richp\Downloads'
$dataRoot = [IO.Path]::GetFullPath((Join-Path (Split-Path -Parent $env:COUNTY_MOVE_SCRIPT) 'data'))
$files = @(
    ,@('SNAPPGrazingDataset_final_062326.xlsx', 'analysis_inputs\ecosystem_services\grazing')
    ,@('Finalized Flagged and Sorted Industry and County Codes.xlsx', 'analysis_inputs\ecosystem_services\jobs')
    ,@('total_valuation_all_regions_fvs.csv', 'analysis_inputs\ecosystem_services\timber')
    ,@('NCPvalue_landings_marine_fish_noaa_2023_state.csv', 'analysis_inputs\ecosystem_services\marine_fisheries')
    ,@('Harvest_table_with_value.csv', 'analysis_inputs\ecosystem_services\inland_fisheries')
    ,@('HEAT_all_tracts_2021.csv', 'analysis_inputs\ecosystem_services\urban_heat')
    ,@('hypertension_linear_us_counties_all.gpkg', 'analysis_inputs\ecosystem_services\physical_health')
    ,@('ucsc_cwon_studyunits.gpkg', 'analysis_inputs\ecosystem_services\mangroves')
    ,@('snapp_air_quality.csv', 'analysis_inputs\ecosystem_services\air_quality')
    ,@('usa_nature_assessment_recreation__snapp_estimates.gpkg', 'analysis_inputs\recreation')
    ,@('counties_ecosystem_services_20260910_130816.gpkg', 'analysis_results\zonal_statistics\counties')
    ,@('counties_coastline_length_20260910_130816.gpkg', 'analysis_results\zonal_statistics\counties')
    ,@('counties_freshwater_area_20260910_130816.csv', 'analysis_results\zonal_statistics\counties')
)
$moved = 0
$skipped = 0
$failed = 0
Write-Host "Destination root: $dataRoot"
if ($dryRun) { Write-Host 'Preview only; no files or folders will be changed.' }
foreach ($entry in $files) {
    try {
        $source = Join-Path $downloads $entry[0]
        $folder = Join-Path $dataRoot $entry[1]
        $destination = [IO.Path]::GetFullPath((Join-Path $folder $entry[0]))
        if (-not $destination.StartsWith($dataRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Destination is outside the data directory: $destination"
        }
        if (Test-Path -LiteralPath $destination) {
            Write-Host "SKIP (destination exists): $destination"
            $skipped++
            continue
        }
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
            Write-Host "SKIP (source missing): $source"
            $skipped++
            continue
        }
        if ($dryRun) {
            Write-Host "WOULD MOVE: $source -> $destination"
        } else {
            New-Item -ItemType Directory -Path $folder -Force | Out-Null
            Move-Item -LiteralPath $source -Destination $destination -ErrorAction Stop
            Write-Host "MOVED: $destination"
            $moved++
        }
    } catch {
        Write-Host "ERROR: $($_.Exception.Message)" -ForegroundColor Red
        $failed++
    }
}
Write-Host "Finished. Moved: $moved; skipped: $skipped; errors: $failed."
if ($failed -gt 0) { exit 1 }
exit 0
