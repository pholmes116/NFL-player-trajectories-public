## Project proposal form

Please provide the information requested in the following form. Try provide concise and informative answers.

**1. What is your project title?**

Generating post-snap player trajectories in American Football (NFL) 

**2. What is the problem that you want to solve?**

Sports analytics has become an integral part of American football, where teams constantly seek data-driven insights to refine their strategies, optimize player performance, and gain a competitive edge. Traditional statistics such as yards gained, or points scored only offer a limited snapshot of on-field events. Modern player tracking data, however, provides a wealth of spatiotemporal information that can be used to examine nuances in player behaviour and team dynamics.  

Despite the growing availability of detailed tracking data, there remains a gap in the ability to anticipate how players will actually move once the ball is “snapped”.1 This project aims to address that gap by building a model that simulates post-snap player trajectories using their pre-snap positions and contextual metadata (e.g., down, distance, formations). Accurately predicting multiple players’ movements in real time is crucial for coaches who want to identify potential mismatches, plan defensive/offensive schemes, and improve in-game decision-making. 

Beyond predicting player trajectories, the embeddings learned by a transformer model in this project can power a host of downstream tasks. By distilling rich contextual information—such as individual player tendencies, situational factors, and team-level strategies—into compact vectors, these embeddings become a powerful representation of on-field dynamics. For instance, analysts could cluster player embeddings to discover hidden role similarities or playing styles, informing draft decisions or match-up analyses. 

**3. What deep learning methodologies do you plan to use in your project?**

Due to the sequential nature of our data, we propose the use of an autoregressive model. More specifically, a Transformer, which to the best of our knowledge, has not been used to predict NFL player movement in academic literature (Only in Blog posts).2  We opted to use transformers as opposed to RNNs such as LSTMs, or GRUs because with sequences of up to 1886 feature vectors in length, we believe the Transformer is better suited at capturing the long-term dependencies in the data. 

1. Model inputs: 
    a. Individual feature vector includes the following information for each player on the field: 
        i. Location  
        ii. Direction player is facing 
        iii. Direction of player movement 
        iv. OHE of position that player plays (QB, WR, DE, etc...) 
        v. Weight of player 
        vi. Height of player  
        vii. ... 
        viii. And any other relevant features at a player, play, game, or team level 
    b. Input Sequence: 
        i. The input sequence will consist of ordered and masked vectors corresponding to all measurements made in a play up to time t. 

2. Model body: 
    a. Embedding layer 
    b. Positional encoding
    c. Masked multi head attention-head 
    d. Norm and residual layer 
    e. Multiheaded attention layer
    f. ....... 
    g. FF Layer 
    h. Output layer (Output dim 23*2 for all players + ball for acceleration along the x and y axis) 
        i. Might use the reparameterization trick for smoother continuous outputs. 
        ii. If this task is deemed too complicated or computationally intensive, we might consider reducing the number players or features we decide to predict over. 

3. Physics-Informed Output transformation: 
    a. Instead of directly predicting positions, the model will output acceleration 
    b. With numerical Integration methods we will integrate acceleration to recover player positions. 

4. Loss: 
    a. MSE of actual vs predicted locations of players in the field (after output transformation). 
    b. Incorporate an additional physics inspired loss component during training to penalize impossible behavior. 

**4. What dataset will you use? Provide information about the dataset, and a URL for the dataset if available. Briefly discuss suitability of the dataset for your problem.**

We will use the dataset provided in the Kaggle challenge competition NFL Big Data Bowl 2025. This is a sport analytics competition run by the USA’s National Football League (NFL), which has had seven editions already.  

The dataset is composed of the following tables with data from the 2022 season:

1. Game data: game-level details (teams, scores, and dates). 
2. Play data: play-level details (down, yards to go, game quarter, which team is on offense and which team is on defense, etc.). 
3. Player data: static player details (height, weight, birthdate, college, position). 
4. Player-play data: player-level statistics for each play (rushing yards, passing yards, interceptions, etc.). 
5. Tracking data: spatiotemporal tracking data for players on the field with a frequency of 10 frames per second (position, speed, orientation, and events at each timestamp). 

The tracking data comes from Neft Gen Stats, an advanced data collection and analysis system used by the National Football League (NFL) to track and measure player performance beyond traditional statistics. The system captures detailed information about player movements, speed, acceleration, and positioning on the field during games. It uses radio frequency sensors strapped to the shoulder pads of the players to capture their position and their upper-body orientation. 

One advantage of this dataset is the fact that it requires no data cleaning. It is already completely standardized. This means that we will not have to resort to external sources, and that our data processing effort will solely involve merging and sub-setting the data to suit our purposes and doing feature engineering. Additionally, with a tracking rate of 10 frames per second, it is incredibly granular, so we have enough data for training purposes. 

**5. List key references (e.g. research papers) that your project will be based on?**

Kaggle data location and inspiration (also contains a data dictionary): 
https://www.kaggle.com/competitions/nfl-big-data-bowl-2025 

Medium blog post (and corresponding GitHub) utilizing transformers for NFL Next Gen Stats geospatial data to better understand NFL games: 
https://medium.com/data-science/transformers-can-generate-nfl-plays-introducing-qb-gpt-2d40f16a03eb 
https://github.com/samchaineau/StratFormer/tree/main 

Simulating Defensive Trajectories in American Football for Predicting League Average Defensive Movements:  
https://www.frontiersin.org/journals/sports-and-active-living/articles/10.3389/fspor.2021.669845/full 

Predicting the results of NFL games using machine learning: 
https://aaltodoc.aalto.fi/items/8a329777-bdfe-4800-8d8d-d6dc3e75be70 

Predicting plays in the National Football League: 
https://journals.sagepub.com/doi/full/10.3233/JSA-190348#ref007 

**Please indicate whether your project proposal is ready for review (Yes/No):**

Yes

## Feedback (to be provided by the course lecturer)
