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

  def recommend_movie(self, uid):
    return random.randint(0, len(self.movies)-1)

  def recommend_movie_with_memory(self, uid):
    if not self.previous_recommendations.has_key(uid):
      self.previous_recommendations[uid] = []

    movies_not_yet_recommended = [movie_id for (movie_id, movie_title) in enumerate(self.movies)
                                  if movie_id not in self.previous_recommendations[uid]]
    print movies_not_yet_recommended
    new_movie_id = -1
    if len(movies_not_yet_recommended) > 0:
      new_movie_id = random.choice(movies_not_yet_recommended)
      self.previous_recommendations[uid].append(new_movie_id)

    return new_movie_id
  
  @mimerender(
    default = 'html',
    html = render_html,
    xml  = render_xml,
    json = render_json,
    txt  = render_txt
  )
  def GET(self, uid):
    recommended_movie = self.recommend_movie(uid)

    message = ''
    if recommended_movie >= 0:
      message = 'You should watch %s tonight' % self.movies[recommended_movie]
    else:
      message = 'No more movies left for you to watch!'

    return {'message': message}

if __name__ == "__main__":
  app.run()
