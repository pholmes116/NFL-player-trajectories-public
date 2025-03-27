import plotly.graph_objects as go
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_dir = 'NFL_data'

# You can inherit these colors from nflverse, this is for completeness/convenience 
colors = {
    'ARI':["#97233F","#000000","#FFB612"],
    'ATL':["#A71930","#000000","#A5ACAF"],
    'BAL':["#241773","#000000"],
    'BUF':["#00338D","#C60C30"],
    'CAR':["#0085CA","#101820","#BFC0BF"],
    'CHI':["#0B162A","#C83803"],
    'CIN':["#FB4F14","#000000"],
    'CLE':["#311D00","#FF3C00"],
    'DAL':["#003594","#041E42","#869397"],
    'DEN':["#FB4F14","#002244"],
    'DET':["#0076B6","#B0B7BC","#000000"],
    'GB' :["#203731","#FFB612"],
    'HOU':["#03202F","#A71930"],
    'IND':["#002C5F","#A2AAAD"],
    'JAX':["#101820","#D7A22A","#9F792C"],
    'KC' :["#E31837","#FFB81C"],
    'LA' :["#003594","#FFA300","#FF8200"],
    'LAC':["#0080C6","#FFC20E","#FFFFFF"],
    'LV' :["#000000","#A5ACAF"],
    'MIA':["#008E97","#FC4C02","#005778"],
    'MIN':["#4F2683","#FFC62F"],
    'NE' :["#002244","#C60C30","#B0B7BC"],
    'NO' :["#101820","#D3BC8D"],
    'NYG':["#0B2265","#A71930","#A5ACAF"],
    'NYJ':["#125740","#000000","#FFFFFF"],
    'PHI':["#004C54","#A5ACAF","#ACC0C6"],
    'PIT':["#FFB612","#101820"],
    'SEA':["#002244","#69BE28","#A5ACAF"],
    'SF' :["#AA0000","#B3995D"],
    'TB' :["#D50A0A","#FF7900","#0A0A08"],
    'TEN':["#0C2340","#4B92DB","#C8102E"],
    'WAS':["#5A1414","#FFB612"],
    'football':["#CBB67C","#663831"]
}

def preprocess_data(tracking_data, players_data):
    """
    merges data for the `animate_play()` function 
    """
    tracking_df = pd.merge(df,players,how="left",on = ["nflId",'displayName'])
    return tracking_df

def hex_to_rgb_array(hex_color):
    """
    take in hex val and return rgb np array
    helper for 'color distance' issues 
    """
    return np.array(tuple(int(hex_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))

def ColorDistance(hex1,hex2):
    """
    d = {} distance between two colors(3)
    helper for 'color distance' issues 
    """
    if hex1 == hex2:
        return 0
    rgb1 = hex_to_rgb_array(hex1)
    rgb2 = hex_to_rgb_array(hex2)
    rm = 0.5*(rgb1[0]+rgb2[0])
    d = abs(sum((2+rm,4,3-rm)*(rgb1-rgb2)**2))**0.5
    return d

def ColorPairs(team1,team2):
    """
    Pairs colors given two teams
    If colors are 'too close' in hue, switch to alt color  
    """
    color_array_1 = colors[team1]
    color_array_2 = colors[team2]
    # If color distance is small enough then flip color order
    if ColorDistance(color_array_1[0],color_array_2[0])<500:
        return {
          team1:[color_array_1[0],color_array_1[1]],
          team2:[color_array_2[1],color_array_2[0]],
          'football':colors['football']
        }
    else:
        return {
          team1:[color_array_1[0],color_array_1[1]],
          team2:[color_array_2[0],color_array_2[1]],
          'football':colors['football']
        }

def animate_play(games,tracking_df,play_df,players,gameId,playId):
    """
    Generates an animated play using the tracking data. 
    """
    selected_game_df = games.loc[games['gameId']==gameId].copy()
    selected_play_df = play_df.loc[(play_df['playId']==playId) & (play_df['gameId']==gameId)].copy()

    tracking_players_df = tracking_df.copy()
    selected_tracking_df = tracking_players_df.loc[(tracking_players_df['playId']==playId)&(tracking_players_df['gameId']==gameId)].copy()

    sorted_frame_list = selected_tracking_df.frameId.unique()
    sorted_frame_list.sort()

    # get good color combos
    team_combos = list(set(selected_tracking_df['club'].unique())-set(['football']))
    color_orders = ColorPairs(team_combos[0],team_combos[1])

    # get play General information
    line_of_scrimmage = selected_play_df['absoluteYardlineNumber'].values[0]

    # Fixing first down marker issue from last year
    if selected_tracking_df['playDirection'].values[0] == 'right':
        first_down_marker = line_of_scrimmage + selected_play_df['yardsToGo'].values[0]
    else:
        first_down_marker = line_of_scrimmage - selected_play_df['yardsToGo'].values[0]
    down = selected_play_df['down'].values[0]
    quarter = selected_play_df['quarter'].values[0]
    gameClock = selected_play_df['gameClock'].values[0]
    playDescription = selected_play_df['playDescription'].values[0]

    # Handle case where we have a really long Play Description and want to split it into two lines
    if len(playDescription.split(" "))>15 and len(playDescription)>115:
        playDescription = " ".join(playDescription.split(" ")[0:16]) + "<br>" + " ".join(playDescription.split(" ")[16:])

    # initialize plotly start and stop buttons for animation
    updatemenus_dict = [
      {
          "buttons": [
              {
                  "args": [None, {"frame": {"duration": 100, "redraw": False},
                              "fromcurrent": True, "transition": {"duration": 0}}],
                  "label": "Play",
                  "method": "animate"
              },
              {
                  "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                  "label": "Pause",
                  "method": "animate"
              }
          ],
          "direction": "left",
          "pad": {"r": 10, "t": 87},
          "showactive": False,
          "type": "buttons",
          "x": 0.1,
          "xanchor": "right",
          "y": 0,
          "yanchor": "top"
      }
    ]
    # initialize plotly slider to show frame position in animation
    sliders_dict = {
      "active": 0,
      "yanchor": "top",
      "xanchor": "left",
      "currentvalue": {
          "font": {"size": 20},
          "prefix": "Frame:",
          "visible": True,
          "xanchor": "right"
      },
      "transition": {"duration": 300, "easing": "cubic-in-out"},
      "pad": {"b": 10, "t": 50},
      "len": 0.9,
      "x": 0.1,
      "y": 0,
      "steps": []
    }


    frames = []
    for frameId in sorted_frame_list:
        data = []
        # Add Numbers to Field
        data.append(
          go.Scatter(
              x=np.arange(20,110,10),
              y=[5]*len(np.arange(20,110,10)),
              mode='text',
              text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
              textfont_size = 30,
              textfont_family = "Courier New, monospace",
              textfont_color = "#ffffff",
              showlegend=False,
              hoverinfo='none'
          )
        )
        data.append(
          go.Scatter(
              x=np.arange(20,110,10),
              y=[53.5-5]*len(np.arange(20,110,10)),
              mode='text',
              text=list(map(str,list(np.arange(20, 61, 10)-10)+list(np.arange(40, 9, -10)))),
              textfont_size = 30,
              textfont_family = "Courier New, monospace",
              textfont_color = "#ffffff",
              showlegend=False,
              hoverinfo='none'
          )
        )
        # Add line of scrimage
        data.append(
          go.Scatter(
              x=[line_of_scrimmage,line_of_scrimmage],
              y=[0,53.5],
              line_dash='dash',
              line_color='blue',
              showlegend=False,
              hoverinfo='none'
          )
        )
        # Add First down line
        data.append(
          go.Scatter(
              x=[first_down_marker,first_down_marker],
              y=[0,53.5],
              line_dash='dash',
              line_color='yellow',
              showlegend=False,
              hoverinfo='none'
          )
        )
        # Add Endzone Colors
        endzoneColors = {0:color_orders[selected_game_df['homeTeamAbbr'].values[0]][0],
                        110:color_orders[selected_game_df['visitorTeamAbbr'].values[0]][0]}
        for x_min in [0,110]:
            data.append(
              go.Scatter(
                  x=[x_min,x_min,x_min+10,x_min+10,x_min],
                  y=[0,53.5,53.5,0,0],
                  fill="toself",
                  fillcolor=endzoneColors[x_min],
                  mode="lines",
                  line=dict(
                      color="white",
                      width=3
                      ),
                  opacity=1,
                  showlegend= False,
                  hoverinfo ="skip"
              )
            )
        # Plot Players
        for team in selected_tracking_df['club'].unique():
            plot_df = selected_tracking_df.loc[(selected_tracking_df['club']==team) & (selected_tracking_df['frameId']==frameId)].copy()

            if team != 'football':
                hover_text_array=[]

                for nflId in plot_df['nflId'].unique():
                    selected_player_df = plot_df.loc[plot_df['nflId']==nflId]
                    nflId = int(selected_player_df['nflId'].values[0])
                    displayName = selected_player_df['displayName'].values[0]
                    s = round(selected_player_df['s'].values[0] * 2.23693629205, 3)
                    text_to_append = f"nflId:{nflId}<br>displayName:{displayName}<br>Player Speed:{s} MPH"
                    hover_text_array.append(text_to_append)

                data.append(go.Scatter(x=plot_df['x'], y=plot_df['y'],
                                      mode = 'markers',
                                      marker=go.scatter.Marker(color=color_orders[team][0],
                                                              line=go.scatter.marker.Line(width=2,
                                                                                          color=color_orders[team][1]),
                                                              size=10),
                                      name=team,hovertext=hover_text_array,hoverinfo='text'))
            else:
                data.append(go.Scatter(x=plot_df['x'], y=plot_df['y'],
                                      mode = 'markers',
                                      marker=go.scatter.Marker(
                                        color=color_orders[team][0],
                                        line=go.scatter.marker.Line(width=2,
                                                                    color=color_orders[team][1]),
                                        size=10),
                                      name=team,hoverinfo='none'))
        # add frame to slider
        slider_step = {'args': [
          [frameId],
          {'frame': {'duration': 100, 'redraw': False},
            'mode': 'immediate',
            'transition': {'duration': 0}}
        ],
          'label': str(frameId),
          'method': 'animate'}
        sliders_dict['steps'].append(slider_step)
        frames.append(go.Frame(data=data, name=str(frameId)))

    scale=10
    layout = go.Layout(
        autosize=False,
        width=120*scale,
        height=60*scale,
        xaxis=dict(range=[0, 120], autorange=False, tickmode='array',tickvals=np.arange(10, 111, 5).tolist(),showticklabels=False),
        yaxis=dict(range=[0, 53.3], autorange=False,showgrid=False,showticklabels=False),

        plot_bgcolor='#00B140',
        # Create title and add play description at the bottom of the chart for better visual appeal
        title=f"GameId: {gameId}, PlayId: {playId}<br>{gameClock} {quarter}Q"+"<br>"*19+f"{playDescription}",
        updatemenus=updatemenus_dict,
        sliders = [sliders_dict]
    )

    fig = go.Figure(
        data=frames[0]['data'],
        layout= layout,
        frames=frames[1:]
    )
    # Create First Down Markers
    for y_val in [0,53]:
        fig.add_annotation(
              x=first_down_marker,
              y=y_val,
              text=str(down),
              showarrow=False,
              font=dict(
                  family="Courier New, monospace",
                  size=16,
                  color="black"
                  ),
              align="center",
              bordercolor="black",
              borderwidth=2,
              borderpad=4,
              bgcolor="#ff7f0e",
              opacity=1
              )
    # Add Team Abbreviations in EndZone's
    for x_min in [0,110]:
        if x_min == 0:
            angle = 270
            teamName=selected_game_df['homeTeamAbbr'].values[0]
        else:
            angle = 90
            teamName=selected_game_df['visitorTeamAbbr'].values[0]
        fig.add_annotation(
          x=x_min+5,
          y=53.5/2,
          text=teamName,
          showarrow=False,
          font=dict(
              family="Courier New, monospace",
              size=32,
              color="White"
              ),
          textangle = angle
        )
    return fig



if __name__ == "__main__":
    games = pd.read_csv(f'{project_dir}/games.csv')
    plays = pd.read_csv(f'{project_dir}/plays.csv')
    players = pd.read_csv(f'{project_dir}/players.csv')
    tracking_df = pd.read_csv(f'{project_dir}/tracking_week_1.csv')

    # random game/play -- should work with any! 
    gid = 2022091113
    pid = 3853 

    fig = animate_play(games=games, tracking_df=tracking_df,
                       play_df=plays, players=players, gameId=gid,
                       playId=pid)
    fig.show()