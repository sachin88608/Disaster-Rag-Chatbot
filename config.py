import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # GROQ Configuration
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    # Embedding Configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    
    # Vector Database Configuration
    CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "disaster_knowledge")
    
    # RAG Configuration
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    TOP_K_RESULTS = 8
    SIMILARITY_THRESHOLD = 0.2
    
    # Supported File Types
    SUPPORTED_EXTENSIONS = {'.pdf', '.csv', '.xlsx', '.xls', '.txt', '.md'}
    
    # Data Sources Configuration
    DATA_SOURCES = {
        'earthquake': {
            'urls': ['https://www.emdat.be',
                    'https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports',
                    'https://datascience.codata.org/articles/10.5334/dsj-2025-008',
                    'https://www.iitk.ac.in/nicee/wcee/article/14_01-1022.pdf',
                    'https://www.eeri.org/projects/earthquake-clearinghouse/',
                    'https://www.globalquakemodel.org/gem',
                    'https://ngawest2.berkeley.edu',
                    'https://ds.iris.edu/ds/nodes/dmc',
                    'https://data.humdata.org',
                    'https://reliefweb.int/disasters',
                    'https://arxiv.org/search/?query=earthquake&searchtype=all&abstracts=show',
                    'https://www.worldbank.org/en/topic/disasterriskmanagement',
                    'https://www.ngdc.noaa.gov/hazel/view/hazards/earthquake/event-data?maxYear=2025&minYear=1990',
                    'https://www.emsc-csem.org',
                    'https://earthquake.usgs.gov/data/shakemap/',
                    'https://gdacs.org',
                    'https://earthquake.usgs.gov/earthquakes/search/',
                    'https://reliefweb.int/disasters?_gl=1*juuxfq*_ga*NzI2NjIxNDEzLjE3NDk3NDE0MjA.*_ga_E60ZNX2F68*czE3NDk3NDE0MjAkbzEkZzEkdDE3NDk3NDE0NDgkajMyJGwwJGgw',
                    'https://reliefweb.int/disaster/fl-2025-000076-bgd',
                    'https://reliefweb.int/disaster/ep-2025-000062-ecu',
                    'https://earthquake.usgs.gov/fdsnws/event/1/',
                    'https://earthquake.usgs.gov/data/comcat/',
                    'https://earthquake.usgs.gov/earthquakes/map/',
                    'https://www.emsc-csem.org/',
                    'https://www.globalquakemodel.org/',
                    'https://platform.openquake.org/',
                    'https://www.ready.gov/earthquakes',
                    'https://ds.iris.edu/seismon/',
                    'https://www.ready.gov/earthquakes',
                    'https://www.gfdrr.org/en/publications',
                    'https://data.humdata.org/organization/gfdrr',
                    'https://www.kaggle.com/datasets/headsortails/us-natural-disaster-declarations',
                    'https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset',
                    'http://kaggle.com/datasets/umeradnaan/prediction-of-disaster-management-in-2024',
                    'https://www.kaggle.com/datasets/rgbnihal/c2a-dataset',
                    'https://www.kaggle.com/datasets/christianlillelund/passenger-list-for-the-estonia-ferry-disaster',
                    'https://www.kaggle.com/datasets/rupakroy/cyclone-wildfire-flood-earthquake-database',
                    'https://www.kaggle.com/datasets/fema/federal-disasters',
                    'http://kaggle.com/datasets/rahultp97/louisiana-flood-2016',
                    'https://www.kaggle.com/datasets/usgs/earthquake-database',
                    'https://www.iris.edu/hq/inclass',
                    'https://earthquake.usgs.gov/research/',
                    'https://reliefweb.int/updates?search=earthquake',
                    'https://sedac.ciesin.columbia.edu/data/collection/ndh',
                    'https://earthdata.nasa.gov/',
                    'https://www.who.int/health-topics/earthquakes#tab=tab_1',
                    'https://huggingface.co/datasets',
                    'https://paperswithcode.com/task/disaster-response',
                    'https://earthquake.usgs.gov/earthquakes/feed/v1.0/geojson.php',
                    'https://docs.google.com/spreadsheets/d/e/2PACX-1vQcVkjSTYsPCzoUuar82a_W4VmRClkzTtisFOrRAX3knhDGSlJoyBhp6ruKlgqqOz5cg7qCjL1Op4YH/pubhtml',
                    'https://data.humdata.org/dataset/?q=flood+disaster&sort=score+desc%2C+last_modified+desc&ext_page_size=25'
                    ],
            'files': ['data/earthquake/afghanistan-natural-disaster-incidents-from-january-to-may-2025.xlsx',
                    'data/earthquake/idmc_internal_displacement_conflict-violence_disasters-39.xlsx',
                    'data/earthquake/ht_climato-hydro-meteo_emdat_data-20220623.xlsx',
                    'data/earthquake/congo-admin1-flood.csv',
                    'data/earthquake/200204_philippines-2019-events-data.xlsx',
                    'data/earthquake/afghanistan-natural-disaster-incidents-from-january-to-september-2022.xlsx',
                    'data/earthquake/afghanistan-natural-disaster-incidents-from-january-to-december-2020.xlsx',
                    'data/earthquake/afghanistan-natural-disaster-incidents-from-january-to-july-2021.xlsx']
        },
        'flood': {
            'urls': ['https://ndem.nrsc.gov.in/documents/downloads/Flood%20Affected%20Area%20%20Atlas%20of%20India%20-Satellite%20based%20study.pdf',
                     'https://data.jrc.ec.europa.eu/dataset/1d128b6c-a4ee-4858-9e34-6210707f3c81',
                     'https://saarc-sdmc.org/sites/default/files/programmes_doc_upload/SDMC_Day2_Lect2_Hands_on_OpenDataPortals.pdf',
                     'https://environment.data.gov.uk/flood-planning/download/cycle-2/',
                     'https://public.emdat.be/',
                    'https://ndmindia.mha.gov.in/ndmi/whatNewSituation',
                    'https://ndmindia.mha.gov.in/ndmi/viewUploadedDocument?uid=NEW2568',
                    'https://ndmindia.mha.gov.in/ndmi/viewUploadedDocument?uid=NEW2567',
                    'https://internal.imd.gov.in/pages/press_release_mausam.php',
                    'https://internal.imd.gov.in/press_release/20250608_pr_4040.pdf',
                    'https://indiawris.gov.in/wiki/doku.php?id=cwc_national_flood_forecasting_network#:~:text=Central%20Water%20Commission%20through%20its,Barrage%20Authorities%2F%20District%20Magistrates%2F%20Sub',
                    'https://rsmcnewdelhi.imd.gov.in/report.php?internal_menu=NjE=',
                    'https://rsmcnewdelhi.imd.gov.in/report.php?internal_menu=NzE=',
                    'https://rsmcnewdelhi.imd.gov.in/report.php?internal_menu=NzI=',
                    'https://disasterscharter.org/web/guest/charter-activations',
                    'https://data.humdata.org/search?q=flood',
                    'https://emergency.copernicus.eu/mapping/ems/floods',
                    'https://www.ndma.gov.in/Natural-Hazards/Floods',
                    'https://www.ndma.gov.in/Natural-Hazards/Cyclone',
                    'https://www.ndma.gov.in/Natural-Hazards/Tsunami',
                    'https://www.ndma.gov.in/Natural-Hazards/Urban-Floods',
                    'https://tsunami.incois.gov.in/TEWS/National.jsp',
                    'https://waterwatch.usgs.gov/index.php?id=ww_flood',
                    'https://waterwatch.usgs.gov/index.php?id=ww_drought',
                    'https://zenodo.org/records/7545697',
                    'https://urs.earthdata.nasa.gov/profile',
                    'https://ndma.gov.in/Natural-Hazards/Urban-Floods',
                    'https://ndma.gov.in/Natural-Hazards/Floods',
                    'https://mausam.imd.gov.in/',
                    'https://data.jrc.ec.europa.eu/collection/id-0054',
                    'https://environment.data.gov.uk/flood-planning/download/cycle-2/'
                    ],
            'files': ['data/RS_Session_255_AU_887.1.csv',
                      'data/flood/public_emdat_data.xlsx',
                      'data/flood/WHO-COVID-19-global-daily-data.csv',
                      'data/flood/28_e80340_TCP21(2024 edition).pdf'
                      'data/flood/61_245057_Cyclone Warning SOP Booklet final.pdf',
                      'data/flood/61_d385f4_FINAL SOP 2024_December.pdf',
                      'data/flood/71_7d2c29_MARINE SOP.pdf',
                      'data/flood/72_211daa_bulletin_sop.pdf',
                      'data/flood/flood_risk_dataset_india.csv',
                      'data/flood/s_fld_haz_ar.xls',
                      'data/flood/District_FloodedArea.csv',
                      'data/flood/District_FloodImpact.csv',
                      'data/flood/India_Flood_Inventory_v3.csv',
                      'data/flood/allocation-partners-un-19july2011.csv',
                      'data/flood/cyberFlood_1104.csv',
                      'data/flood/Floodplains_Outline.csv',
                      'data/flood/flood.csv']
        },
        'Covid': {
            'urls': ['https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports',
                     'https://covid19dashboard.mohfw.gov.in/pdf/abc.pdf',
                     'https://covid19dashboard.mohfw.gov.in/pdf/def.pdf',
                     'https://covid19dashboard.mohfw.gov.in/pdf/CummulativeCovidVaccinationReport19Jun2024.pdf',
                     'https://www.kaggle.com/datasets/imdevskp/corona-virus-report',
                     'https://www.mohfw.gov.in/',
                     'https://data.who.int/dashboards/covid19/cases?n=c',
                     'https://www.kaggle.com/datasets/imdevskp/corona-virus-report',
                     'https://covid19datahub.io/',
                     'https://ourworldindata.org/coronavirus',
                     'https://www.cdc.gov/coronavirus/2019-ncov/index.html'
                     ],
            'files': ['data/Covid/country_wise_latest.csv']
        },
        'tsunami': {
            'urls': ['https://www.ngdc.noaa.gov/hazel/view/hazards/tsunami/event-data?maxYear=2025&minYear=1900',
                    'https://cbe.miis.edu/cgi/viewcontent.cgi?article=1091&context=booc',
                    'https://www.weather.gov/itic-car/tsunami_awareness_materials',
                    'https://tsunami.ioc.unesco.org/en/pacific/itic',
                    'https://psmsl.org/data/obtaining/',
                    'https://www.ncei.noaa.gov/products/natural-hazards',
                    'https://www.tsunami.gov/',
                    'https://www.dnr.wa.gov/programs-and-services/geology/geologic-hazards/tsunamis#understanding-tsunamis.1',
                    'https://www.dnr.wa.gov/programs-and-services/geology/geologic-hazards/tsunamis#tsunamis-in-washington',
                    'https://www.dnr.wa.gov/programs-and-services/geology/geologic-hazards/tsunamis#preparation-and-evacuation.4',
                    'https://www.dnr.wa.gov/programs-and-services/geology/geologic-hazards/tsunamis#tsunami-evacuation-maps',
                    'https://www.dnr.wa.gov/programs-and-services/geology/geologic-hazards/tsunamis#tsunami-hazard-maps',
                    'https://www.dnr.wa.gov/programs-and-services/geology/geologic-hazards/tsunamis#tsunami-simulation-videos',
                    'https://www.dnr.wa.gov/programs-and-services/geology/geologic-hazards/tsunamis#tsunami-alerts',
                    'https://www.dnr.wa.gov/programs-and-services/geology/geologic-hazards/tsunamis#historical-tsunamis-worldwide',
                    'https://www.dnr.wa.gov/programs-and-services/geology/geologic-hazards/tsunamis#building-code-design-zone-map',
                    'https://reliefweb.int/disaster/ts-2004-000147-idn',
                    'https://catalog.data.gov/dataset/ncei-wds-global-historical-tsunami-database-2100-bc-to-present1#:~:text=The%20Global%20Historical%20Tsunami%20Database,database%20includes%20two%20related%20files.',
                    'https://appliedsciences.nasa.gov/what-we-do/disasters/tsunamis',
                    'https://tsunami.org/article-tsunami-and-science/',
                    'https://www.ncei.noaa.gov/products/natural-hazards/tsunamis-earthquakes-volcanoes/tsunamis',
                    'https://tsunami.ioc.unesco.org/en/pacific/itic',
                    'https://incois.gov.in/',
                    'http://www.bom.gov.au/tsunami/',
                    'https://geofon.gfz-potsdam.de/',
                    'https://www.britannica.com/science/tsunami',
                    'https://www.bbc.com/news/topics/c70467809wzt',
                    'https://www.bbc.com/news/articles/c4g3k3pr9e0o',
                    'https://legacy.itic.ioc-unesco.org/legacy.itic.ioc-unesco.org/index464f.html?option=com_content&view=category&layout=blog&id=2767&Itemid=3406',
                    'https://www.livescience.com/planet-earth/tsunami/tsunamis-up-to-90-feet-high-smash-into-new-zealand-every-580-years-study-finds',
                    'https://www.livescience.com/planet-earth/tsunami/1st-mega-tsunami-on-record-since-antiquity-was-triggered-by-tonga-volcanic-eruption',
                    'https://www.ncei.noaa.gov/products/natural-hazards/tsunamis-earthquakes-volcanoes/tsunamis/global-historical-data',
                    'https://www.ngdc.noaa.gov/hazel/view/hazards/tsunami/event-data',
                    'https://rmets.onlinelibrary.wiley.com/doi/10.1002/gdj3.270',
                    'https://tsunami.incois.gov.in/TEWS/tsunamimodeling.jsp'
                     ],
            'files': ['data/tsunami/GLOBAL_SITES_TD.csv',
                      'data/tsunami/tsunami_dataset.csv']
        }
    }
    
    # Future disaster types (ready for expansion)
    FUTURE_DISASTERS = ['drought', 'covid', 'wildfire', 'hurricane', 'tsunami', 'plane_crash']