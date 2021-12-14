import VueRouter from 'vue-router';

//Pages
import FacePage from './components/pages/FacePage'


const routes = [
  { 
    path: '/', 
    redirect: {
      name: "FacePage"
    }
  },
  {
    path: '/face',
    name: 'FacePage',
    component: FacePage
  },
];

const router = new VueRouter({
  routes,
  mode: 'history',
  linkActiveClass: "is-active"
});

export default router