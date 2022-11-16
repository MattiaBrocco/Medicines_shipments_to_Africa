import os
import sys
import numpy as np
import pandas as pd
import geopy.distance as geod

def cleaning_input(data):
    """
    Parameters
    ----------
    
    data : pd.DataFrame
    
    Fix some columns' values (introducing NaN for given text values,
    other data input, etc.)
    """
    
    
    # NaN inputting
    data["PQ First Sent to Client Date"] = data["PQ First Sent to Client Date"]\
                                           .apply(lambda s: np.nan if s == "Date Not Captured"
                                                  else s)
    data["PO Sent to Vendor Date"] = data["PQ First Sent to Client Date"]\
                                     .apply(lambda s: np.nan if s in ["Date Not Captured",
                                                                      "N/A - From RDC"]
                                            else s)
    data["Weight (Kilograms)"] = data["Weight (Kilograms)"]\
                                 .apply(lambda s: np.nan if s == "Weight Captured Separately"
                                        else s)
    # Fix weight
    data["Weight (Kilograms)"] = data["Weight (Kilograms)"].apply(lambda v: v if not isinstance(v, str)
                                                                  or v.isnumeric() else
                                                                  data.loc[(data["ASN/DN #"] == v.split(" ")[1]) &
                                                                           (data["ID"] == int(v.split(":")[-1].strip("()"))),
                                                                           "Weight (Kilograms)"].values[0])
    data["Weight (Kilograms)"] = data["Weight (Kilograms)"].astype(float)
    ### Fill some (~30 rows) NaN with weight of same item ('ASN/DN #')
    wgt_pivot = data.pivot_table(index = "ASN/DN #", values = "Weight (Kilograms)",
                                 aggfunc = np.nanmedian).reset_index()\
                                 .rename(columns = {"Weight (Kilograms)": "Median KG"})
    data = data.merge(wgt_pivot, on = "ASN/DN #", how = "left")
    data["Weight (Kilograms)"] = data["Weight (Kilograms)"].fillna(data["Median KG"])
    data = data.drop("Median KG", axis = 1)
    
    
    # Fix Freight Cost
    data["Freight Cost (USD)"] = data["Freight Cost (USD)"].apply(lambda s: 0
                                                                  if s == "Freight Included in Commodity Cost"
                                                                  else np.nan if s == "Invoiced Separately" else s)
    data["Freight Cost (USD)"] = data["Freight Cost (USD)"].apply(lambda v:
                                                                  data.loc[(data["ASN/DN #"] == v.split(" ")[1]) &
                                                                           (data["ID"] == int(v.split(":")[-1].strip("()"))),
                                                                           "Freight Cost (USD)"].values[0]
                                                                  if isinstance(v, str) and "See" in v
                                                                  else v).astype(float)
    # Fix Subclassification
    #data["Sub Classification"] = data["Sub Classification"].apply(lambda s: s.split(" - ")[0]
    #                                                              if "Ancillary" in s else s)
    
    #### FEATURE ENGINEERING
    # Dates columns
    for dcol in ["Scheduled Delivery Date", "Delivered to Client Date",
                 "Delivery Recorded Date"]:
        data[dcol] = pd.to_datetime(data[dcol])

    data["Scheduled Delivery YEAR"] = data["Scheduled Delivery Date"].dt.year
    data["Scheduled Delivery MONTH"] = data["Scheduled Delivery Date"].dt.month
    data["Scheduled Delivery DAY"] = data["Scheduled Delivery Date"].dt.day

    # Delays
    data["Delay for Customer"] = (data["Delivered to Client Date"] -\
                                  data["Scheduled Delivery Date"]).dt.days.astype("int64")
    data["Delay in Recording"] = (data["Delivery Recorded Date"] -\
                                  data["Delivered to Client Date"]).dt.days.astype("int64")

    # Fill rows where Shipment Mode with the most common value (Air)s
    data["Shipment Mode"] = data["Shipment Mode"].fillna(data["Shipment Mode"].mode().values[0])   
    
    ### Geographic data

    # Manufacturing Site
    man_site = pd.read_table("man_site.txt", sep = "|")
    dms = dict(zip(man_site["Site"],
                   man_site[["Latitude", "Longitude"]].values.tolist()))

    data["Manufacturing Site Coords"] = data["Manufacturing Site"].map(dms)

    # Country
    coun_coord = pd.read_table("country_loc.txt", sep = "|")
    dcc = dict(zip(coun_coord["Country"],
                   coun_coord[["Latitude", "Longitude"]].values.tolist()))

    data["Country Coords"] = data["Country"].map(dcc)

    # New col: Distance
    data["Distance Manuf Dest"] = data.apply(lambda row:
                                             geod.distance(row["Country Coords"],
                                                           row["Manufacturing Site Coords"]).km,
                                             axis = 1)
    
    # If 'PMO - US Managed' is highly unbalanced, drop it
    if sum(data["Managed By"].value_counts()/len(data) > .99) > 0:
        data = data.drop("Managed By", axis = 1)
    
    return data



def preprocessing_for_ml(data_in):
    
    data = data_in.copy()
    
    ### Dummy Encoding
    # Shipment Mode
    data = pd.concat([data.drop("Shipment Mode", axis = 1),
                      pd.get_dummies(data["Shipment Mode"], drop_first = True)], axis = 1)
    # Fulfill Via
    data = pd.concat([data.drop("Fulfill Via", axis = 1),
                      pd.get_dummies(data["Fulfill Via"], drop_first = True)], axis = 1)

    # Managed by
    if "Managed By" in data.columns:
        data["Managed By"] = data["Managed By"].apply(lambda s: 1 if s == "PMO - US" else 0)
        data = data.rename(columns = {"Managed By": "PMO - US Managed"})

    # First Line Designation
    data["First Line Designation"] = data["First Line Designation"].map({"Yes": 1, "No": 0})

    # Dosage Form
    # NOTE: grouping is done without any prior domain knowledge
    data["Dosage Form"] = data["Dosage Form"].apply(lambda s: "Tablet" if "tablet" in s.lower()
                                                    else "Powder" if "powder" in s.lower()
                                                    else "Capsule" if "capsule" in s.lower()
                                                    else "Test kit" if "test kit" in s.lower()
                                                    else s)
    data = pd.concat([data.drop("Dosage Form", axis = 1),
                      pd.get_dummies(data["Dosage Form"], drop_first = True)], axis = 1)

    # Sub Classification
    data = pd.concat([data.drop("Sub Classification", axis = 1),
                      pd.get_dummies(data["Sub Classification"], drop_first = True)], axis = 1)
    
    # DROP ALL NON-NUMERIC COLUMNS
    # Related to geographic data
    data = data.drop(["Manufacturing Site", "Country",
                      "Manufacturing Site Coords", "Country Coords"], axis = 1)
    # Time data
    data = data.drop(["Scheduled Delivery Date", "Delivered to Client Date",
                      "Delivery Recorded Date"], axis = 1)
    # Drop some columns (mainly textual attributes of the transaction/prodcut)
    # 'Line Item Insurance (USD)'' is removed as it is highly
    # correlated with 'Line Item Value' (r = .96)
    # 'Product Group' is highly correlated (via contingency table)
    # with 'Sub Classification'
    data = data.drop(["ID", "Item Description", "PQ #", "ASN/DN #", "PO / SO #", "Dosage",
                      "Line Item Insurance (USD)", "Molecule/Test Type", "Project Code",
                      "PQ First Sent to Client Date", "PO Sent to Vendor Date",
                      "Product Group",
                      # Text columns
                      "Vendor INCO Term", "Vendor", "Brand"], axis = 1)    
    
    # DROP ALL ROWS WITH NAN VALUES
    data = data.dropna().reset_index(drop = True)
    
    return data


def preprocessing_for_classification(data_in):
    """
    Same function as 'preprocessing_for_ml', without
    the dummy encoding of 'Shipment Mode'.
    """
    data = data_in.copy()
    
    ### Dummy Encoding
    # Fulfill Via
    data = pd.concat([data.drop("Fulfill Via", axis = 1),
                      pd.get_dummies(data["Fulfill Via"], drop_first = True)], axis = 1)

    # Managed by
    if "Managed By" in data.columns:
        data["Managed By"] = data["Managed By"].apply(lambda s: 1 if s == "PMO - US" else 0)
        data = data.rename(columns = {"Managed By": "PMO - US Managed"})

    # First Line Designation
    data["First Line Designation"] = data["First Line Designation"].map({"Yes": 1, "No": 0})

    # Dosage Form
    # NOTE: grouping is done without any prior domain knowledge
    data["Dosage Form"] = data["Dosage Form"].apply(lambda s: "Tablet" if "tablet" in s.lower()
                                                    else "Powder" if "powder" in s.lower()
                                                    else "Capsule" if "capsule" in s.lower()
                                                    else "Test kit" if "test kit" in s.lower()
                                                    else s)
    data = pd.concat([data.drop("Dosage Form", axis = 1),
                      pd.get_dummies(data["Dosage Form"], drop_first = True)], axis = 1)

    # Sub Classification
    data = pd.concat([data.drop("Sub Classification", axis = 1),
                      pd.get_dummies(data["Sub Classification"], drop_first = True)], axis = 1)
    
    # DROP ALL NON-NUMERIC COLUMNS
    # Related to geographic data
    data = data.drop(["Manufacturing Site", "Country",
                      "Manufacturing Site Coords", "Country Coords"], axis = 1)
    # Time data
    data = data.drop(["Scheduled Delivery Date", "Delivered to Client Date",
                      "Delivery Recorded Date"], axis = 1)
    # Drop some columns (mainly textual attributes of the transaction/prodcut)
    # 'Line Item Insurance (USD)'' is removed as it is highly
    # correlated with 'Line Item Value' (r = .96)
    # 'Product Group' is highly correlated (via contingency table)
    # with 'Sub Classification'
    data = data.drop(["ID", "Item Description", "PQ #", "ASN/DN #", "PO / SO #", "Dosage",
                      "Line Item Insurance (USD)", "Molecule/Test Type", "Project Code",
                      "PQ First Sent to Client Date", "PO Sent to Vendor Date",
                      "Product Group",
                      # Text columns
                      "Vendor INCO Term", "Vendor", "Brand"], axis = 1)    
    
    # DROP ALL ROWS WITH NAN VALUES
    data = data.dropna().reset_index(drop = True)
    
    return data