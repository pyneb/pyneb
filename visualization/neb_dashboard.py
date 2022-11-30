import dash
import pandas as pd
import numpy as np
import sys, os
import h5py

pynebDir = os.path.expanduser("~/Research/pyneb/src")
if pynebDir not in sys.path:
    sys.path.insert(0,pynebDir)
import pyneb

from dash.dependencies import Input, Output, State, ALL
dcc = dash.dcc
html = dash.html
dash_table = dash.dash_table
import dash_bootstrap_components as dbc

import plotly
import plotly.express as px
import plotly.graph_objects as go
import json

from scipy import interpolate

import itertools

#%% Modify for different problems
collectiveCoords = ["Q20","Q30",]
logDir = "logs/mass/"

def preprocess_pes():
    originalDf = pd.read_csv("PES.dat",sep="\t")
    # originalDf = originalDf.drop(columns=collectiveCoords) #Uncomment if raw coordinates are present
    originalDf = originalDf.rename(columns={"expected_"+c:c for c in collectiveCoords})
    
    uniqueCoords = [np.unique(originalDf[c]) for c in collectiveCoords]
    mesh = np.meshgrid(*uniqueCoords)
    expectedCoords = np.array(list(itertools.product(*uniqueCoords)))
    
    df = pd.DataFrame(expectedCoords,columns=collectiveCoords)
    df = pd.merge(df,originalDf,how="outer",on=collectiveCoords)
    df = df.sort_values(collectiveCoords)
    
    colsToInterp = list(set(df.columns)-set(collectiveCoords))
    noNanDf = originalDf.copy().dropna()
    fillInterpolator = interpolate.RBFInterpolator(noNanDf[collectiveCoords],
                                                   noNanDf[colsToInterp],
                                                   neighbors=100)
    indsToFix = df[df.isna().any(axis=1)].index
    coordsToFix = df.loc[indsToFix][collectiveCoords]
    df.loc[indsToFix,colsToInterp] = fillInterpolator(coordsToFix)
    
    shp = [len(u) for u in uniqueCoords]
    zz = df["EHFB"].to_numpy().reshape(shp)
    gsInds = pyneb.SurfaceUtils.find_local_minimum(zz,searchPerc=len(collectiveCoords)*[0.25,])
    gsLoc = [np.swapaxes(m,0,1)[gsInds] for m in mesh]
    
    df["EHFB"] -= zz[gsInds]
    return mesh, zz, gsLoc

mesh, zz, gsLoc = preprocess_pes()

externalStyles = [dbc.themes.BOOTSTRAP,'https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=externalStyles,
                suppress_callback_exceptions=True)
app.title = "NEB Dashboard"

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

colors = {
    "graphBackground": "#F5F5F5",
    "background": "#ffffff",
    "text": "#000000"
}

def plot_layout(nm):
    return dbc.Row([
        html.Div(
            id=nm+"-plot-with-slider-container",
            children=[
                dcc.Interval(
                    id=nm+"-plot-interval",
                    disabled=True
                    ),
                html.Div(
                    id=nm+"-plot-with-slider",
                    ),
                dcc.Slider(
                    min=0,
                    max=0,
                    value=0,
                    id=nm+"-slider",
                    ),
                html.Button(
                    "Play/Pause",
                    id=nm+"-play"
                    ),
                ],
            hidden=True
            ),
        
        html.Div(
            id=nm+"-plot-without-slider",
            hidden=True,
            )
        ])

#%% Layout
app.layout = dbc.Container([
    html.H1("NEB Dashboard"),
    
    html.Div([
        dcc.Dropdown(id="available-files",
                     placeholder="Select file:",
                     options=[{"label":key,"value":key} for key in 
                              sorted([f for f in os.listdir(logDir) if f.endswith(".lap")])]
                     )
        ]),
    
    dbc.Row([
        dbc.Col(
            id="neb-params-table",
            children=[],
            width="auto",
            ),
        dbc.Col(
            id="opt-params-table",
            children=[],
            width="auto",
            )
        ],
        # justify="center"
        ),
    
    html.Hr(),
    
    plot_layout("point"),
    
    html.Hr(),
    
    plot_layout("3d"),
    
    html.Hr(),
    
    plot_layout("tangents"),
    
    html.Hr(),
    
    plot_layout("springForce"),
    
    html.Hr(),
    
    plot_layout("netForce"),
    
    html.Hr(),
    
    ],
    
    fluid=True
    )

@app.callback(
    [
     Output("neb-params-table","children"),
     Output("opt-params-table","children"),
     ],
    [
     Input("available-files","value")
     ]
    )
def display_param_table(selectedFile):
    if selectedFile is None:
        return [], []#, True
        
    nebChildren = [html.Div(["NEB Parameters"])]
    log = pyneb.fileio.LoadForceLogger(os.path.join(logDir,selectedFile))
    
    nebParams = []
    for key in ["constraintEneg","k","kappa"]:
        nebParams.append({"Parameter":key,"Value":log.nebParams[key]})
    try:
        for key in ["endpointSpringForce","endpointHarmonicForce"]:
            nebParams.append({"Parameter":key+" (Left)","Value":log.nebParams[key][0]})
            nebParams.append({"Parameter":key+" (Right)","Value":log.nebParams[key][1]})
    except KeyError:
        pass
        
    nebChildren.append(dash_table.DataTable(nebParams,fill_width=False))    
    
    optChildren = [html.Div(["Optimization Parameters"])]
    
    optParams = []
    methodFormatDict = {"verlet_params":"Verlet","fire_params":"FIRE"}
    for key in ["verlet_params","fire_params"]:
        if key in log.__dict__.keys():
            optParams.append({"Parameter":"Method","Value":methodFormatDict[key]})
            for (nm,val) in getattr(log,key).items():
                if nm == "maxmove":
                    for i in range(len(val)):
                        optParams.append({"Parameter":nm+" (Coord. "+str(i)+")","Value":val[i]})
                else:
                    optParams.append({"Parameter":nm,"Value":val})
    for key in ["mass","target_func","target_func_grad"]:
        try:
            optParams.append({"Parameter":key,"Value":getattr(log,key)})
        except AttributeError:
            continue
            
    optChildren.append(dash_table.DataTable(optParams,fill_width=False))
    
    return nebChildren, optChildren#, False

#%% General utilities
def _set_slider_spacing(points):
    return points.shape[0]//50 + 1 #Maybe will work....?

def make_visible(nm):
    @app.callback(
        [
          Output(nm+"-plot-with-slider-container","hidden"),
          Output(nm+"-plot-without-slider","hidden")
        ],
        [
          Input("available-files","value")
          ]
        )
    def callback(selectedFile):
        if selectedFile is None:
            return True, True
        
        log = pyneb.fileio.LoadForceLogger(os.path.join(logDir,selectedFile))
        if log.points.ndim == 3:
            return False, True
        else:
            return True, False
        
    return callback

def set_slider_props(nm):
    @app.callback(
        [
          Output(nm+"-slider","max"),
          Output(nm+"-slider","step")
          ],
        [
          Input("available-files","value"),
          Input(nm+"-plot-with-slider-container","hidden"),
          ]
        )
    def callback(selectedFile,isHidden):
        if isHidden:
            return 0, 0
        
        log = pyneb.fileio.LoadForceLogger(os.path.join(logDir,selectedFile))
        return log.points.shape[0]-1, _set_slider_spacing(log.points)
    
    return callback

def play_plot(nm):
    @app.callback(
        [
         Output(nm+"-plot-interval","disabled"),
         ],
        [
         Input(nm+"-play","n_clicks"),
         ],
        [
         State(nm+"-plot-interval","disabled"),
         ]
        )
    def callback(nClicks,isDisabled):
        if nClicks:
            return not isDisabled, 
        
        return isDisabled, 
    
    return callback

def plot_without_slider(nm,plot_method,dsetToPlot,plotName):
    #TODO: technically still untested
    assert dsetToPlot in ["points","springForce","tangents","netForce",]
    
    @app.callback(
        [
          Output(nm+"-plot-without-slider","children"),
          ],
        [
          Input("available-files","value"),
          Input(nm+"-plot-without-slider","hidden")
          ],
        )
    def callback(selectedFile,isHidden):
        if isHidden:
            return [], 
        
        children = [html.Div([plotName])]
        log = pyneb.fileio.LoadForceLogger(os.path.join(logDir,selectedFile))
        data = getattr(log,dsetToPlot)
        isPlottingPath = False
        
        if dsetToPlot == "points":
            data = data[:,1:]
            isPlottingPath = True
        q2 = log.points[:,0]
        
        #All plot_method functions to be defined to work with/without a slider.
        #Also must return a dcc.Graph object
        fig = plot_method(data.reshape((1,)+data.shape),
                          q2.reshape((1,)+q2.shape),
                          isPlottingPath)
        
        children.append(fig)
        
        return children,
    
    return callback
    
def plot_with_slider(nm,plot_method,dsetToPlot,plotName):
    assert dsetToPlot in ["points","springForce","tangents","netForce",]
    
    #Callback args can be generalized using some pattern-matching something-or-other
    @app.callback(
        [
          Output(nm+"-plot-with-slider","children"),
          Output(nm+"-slider","value"),
          Output(nm+"-plot-interval","n_intervals")
          ],
        [
          Input("available-files","value"),
          Input(nm+"-plot-with-slider-container","hidden"),
          Input(nm+"-slider","value"),
          Input(nm+"-plot-interval","n_intervals"),
          Input(nm+"-slider","step")
          ],
        [
          State(nm+"-plot-interval","disabled")
          ]
        )
    def callback(selectedFile,isHidden,sliderVal,nIntervals,step,
                 playIsDisabled):
        if isHidden:
            return [], sliderVal, 0
        
        children = [html.Div([plotName])]
        log = pyneb.fileio.LoadForceLogger(os.path.join(logDir,selectedFile))
        data = getattr(log,dsetToPlot)
        isPlottingPath = False
        
        if dsetToPlot == "points":
            data = data[:,:,1:]
            isPlottingPath = True
        q2 = log.points[:,:,0]
        
        if not playIsDisabled:
            sliderVal += step
            if sliderVal > data.shape[0]:
                sliderVal = 0
        
        fig = plot_method(data,q2,isPlottingPath,sliderVal)
        
        children.append(fig)
        
        return children, sliderVal, 0

#%% Plotting
def _plot_components_on_path(data,q2,isPlottingPath,idx=0):
    nCoords = data.shape[-1]
    
    q2Min, q2Max = q2.min(), q2.max()
    minVals = [data[:,:,i].min() for i in range(nCoords)]
    maxVals = [data[:,:,i].max() for i in range(nCoords)]
    
    fig = plotly.subplots.make_subplots(rows=nCoords,cols=1)
    
    plotTitles = (nCoords-1)*["",]+["Q20",]
    if isPlottingPath:
        ylabels = collectiveCoords[1:]
    else:
        ylabels = collectiveCoords
    
    for i in range(nCoords):
        fig.add_trace(
            go.Scatter(
                x=q2[idx],y=data[idx,:,i],
                mode="lines+markers",
                line={"color":"black"},
                ),
                row=i+1,col=1, #Row indexing starts at 1 for some reason
            )
        fig.update_xaxes(title_text=plotTitles[i],row=i+1,col=1,range=[q2Min,q2Max])
        fig.update_yaxes(title_text=ylabels[i],row=i+1,col=1,
                          range=[minVals[i],maxVals[i]])
    fig.update_layout(showlegend=False)
        
    return dcc.Graph(figure=fig)

def _plot_path_2d(data,q2,isPlottingPath,idx=0):
    fig = go.Figure(
        data=go.Contour(x=np.unique(mesh[0]),y=np.unique(mesh[1]),z=zz.T,
                        colorscale="Spectral_r",
                        ncontours=30,
                        zmin=-5,
                        zmax=30
                        )
        )
    
    fig.update_xaxes(title_text=collectiveCoords[0])
    fig.update_yaxes(title_text=collectiveCoords[1])
    
    fig.add_trace(
        go.Scatter(
            x=q2[idx],y=data[idx].flatten(),
            mode="lines+markers",
            line={"color":"black"},
            )
        )
    
    fig.update_layout()
    fig.update_traces()
    
    return dcc.Graph(figure=fig)

def _plot_path_3d(data,q2,isPlottingPath,idx=0):
    zzFlipped = np.swapaxes(zz,0,1)
    
    fig = go.Figure(
        data=go.Isosurface(
            x=mesh[0].flatten(),
            y=mesh[1].flatten(),
            z=mesh[2].flatten(),
            value=zzFlipped.flatten(),
            isomin=-10,
            isomax=30,
            surface_count=8,
            # opacity=0.8,
            colorscale="Spectral_r",
            caps=dict(x_show=False, y_show=False)
            )
        )
    
    linePlot = px.line_3d(
        x=q2[idx],y=data[idx,:,0],z=data[idx,:,1],
        )
    linePlot.update_traces(line=dict(color="Black", width=5))
    
    fig.add_traces(
        list(linePlot.select_traces()
            )
        )
    
    camera = dict(
        # up=dict(x=0.2, y=0, z=1),
        # center=dict(x=0, y=0, z=0),
        eye=dict(x=0.5, y=-2.5, z=0.5)
    )
    
    fig.update_layout(scene_camera=camera)
    
    return dcc.Graph(figure=fig,style={'width': '90vh', 'height': '90vh'})

# @app.callback(
#     [
#       Output("point-plot-without-slider","children"),
#       ],
#     [
#       Input("available-files","value"),
#       Input("point-plot-without-slider","hidden")
#       ],
#     )
# def plot_points_without_slider(selectedFile,isHidden):
#     if isHidden:
#         return [], 
    
#     children = [html.Div(["Path"])]
#     log = pyneb.fileio.LoadForceLogger(os.path.join(logDir,selectedFile))
#     points = log.points
    
#     fig = _plot_path_components(points.reshape((1,)+points.shape))
    
#     children.append(dcc.Graph(figure=fig))
    
#     return children,

#%% Making callbacks
if zz.ndim == 2:
    path_plot_func = _plot_path_2d
elif zz.ndim == 3:
    path_plot_func = _plot_components_on_path
    
    make_visible("3d")
    set_slider_props("3d")
    play_plot("3d")
    plot_without_slider("3d",_plot_path_3d,"points","Path")
    plot_with_slider("3d",_plot_path_3d,"points","Path")
else:
    path_plot_func = _plot_components_on_path

make_visible("point")
set_slider_props("point")
play_plot("point")
plot_without_slider("point",path_plot_func,"points","Path")
plot_with_slider("point",path_plot_func,"points","Path")

make_visible("tangents")
set_slider_props("tangents")
play_plot("tangents")
plot_without_slider("tangents",_plot_components_on_path,"tangents","Tangents")
plot_with_slider("tangents",_plot_components_on_path,"tangents","Tangents")

make_visible("springForce")
set_slider_props("springForce")
play_plot("springForce")
plot_without_slider("springForce",_plot_components_on_path,"springForce","Spring Force")
plot_with_slider("springForce",_plot_components_on_path,"springForce","Spring Force")

make_visible("netForce")
set_slider_props("netForce")
play_plot("netForce")
plot_without_slider("netForce",_plot_components_on_path,"netForce","Net Force")
plot_with_slider("netForce",_plot_components_on_path,"netForce","Net Force")

if __name__ == '__main__':
    app.run_server(debug=True)
