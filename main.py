import json
import random
import web
from mimerender import mimerender

render_xml = lambda message: '<message>%s</message>'%message
render_json = lambda **args: json.dumps(args)
render_html = lambda message: '<html><body>%s</body></html>'%message
render_txt = lambda message: message

urls = (
  '/users/(\d+)/recommendations', 'recommendations'
)
app = web.application(urls, globals())

class recommendations:
  movies = ['Hugo',
            'Tower Heist',
            'J. Edgar',
            'Moneyball',
            'The Help',
            'Johnny English Reborn',
            'Justice League: Doom',
            'The Twilight Saga: Breaking Dawn - Part 1',
            'Drive',
            'Midnight in Paris',
           ]
  previous_recommendations = {}


  def recommend_movies(self, uid, count):
    recommended_movies = []
    if count >= len(self.movies):
      recommended_movies = range(len(self.movies))
    else:
      recommended_movies = random.sample(range(len(self.movies)), count)

    return recommended_movies


  def recommend_movies_with_memory(self, uid, count):
    if not self.previous_recommendations.has_key(uid):
      self.previous_recommendations[uid] = []

    movies_not_yet_recommended = [movie_id for (movie_id, movie_title) in enumerate(self.movies)
                                  if movie_id not in self.previous_recommendations[uid]]

    recommended_movies = []
    if count >= len(movies_not_yet_recommended):
      recommended_movies = movies_not_yet_recommended
    else:
      recommended_movies = random.sample(movies_not_yet_recommended, count)

    self.previous_recommendations[uid] += recommended_movies
    return recommended_movies

  
  @mimerender(
    default = 'html',
    html = render_html,
    xml  = render_xml,
    json = render_json,
    txt  = render_txt
  )
  def GET(self, uid):
    user_data = web.input(count="1")
    recommended_movies = self.recommend_movies(uid, int(user_data.count))
    message = [self.movies[i] for i in recommended_movies]
    return {'message': message}

if __name__ == "__main__":
  app.run()
